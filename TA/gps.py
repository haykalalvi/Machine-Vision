#!/usr/bin/env python3
"""
Integrated Noise Monitoring System
ESP32-S3 (audio via USB) + GPS NEO-7M (location via GPIO UART)

=============================================================================
ARCHITECTURE
=============================================================================

  [ESP32-S3] ──USB CDC──► [SerialReaderProcess]
                                  │ raw_queue (mp.Queue)
                                  ▼
                        [OctaveAnalyzerProcess]
                                  │ result_queue (mp.Queue)
                                  ▼
                          [PublisherProcess] ◄── gps_location (mp.Array)
                                                        ▲
                                              [GpsReaderProcess]
                                           (GPIO UART /dev/ttyAMA0)

=============================================================================
GPS INTEGRATION DESIGN
=============================================================================

  WHY NO SYNCHRONIZATION IS NEEDED:
    The device is stationary or slow-moving. GPS location changes on the
    order of meters per second at most. Audio packets arrive every 125ms.
    There is no meaningful difference between "GPS fix taken at 12:00:00.000"
    and "GPS fix taken at 12:00:00.125". A shared mp.Array with a lock is
    sufficient — no queues, no timestamps, no sync barriers.

  GPS POLLING INTERVAL (GPS_UPDATE_INTERVAL_SEC):
    Set to 300s (5 min) by default. The GPS process reads the UART for a
    short burst every N minutes, parses the first valid GGA sentence, then
    sleeps. Between updates, every published result uses the last known fix.
    This means the GPS LED will blink only briefly every 5 minutes — normal.

  LAST-KNOWN-LOCATION ASSUMPTION:
    If no valid GPS fix has been received yet (e.g. device just powered on
    and still acquiring), lat/lon are published as 0.0. The publisher logs
    a warning in that case. Once a fix is received, it persists until the
    next update.

=============================================================================
PACKET LAYOUT (must match usb_packet_t in ESP32 firmware):
    start_marker             : uint32   4 B
    timestamp_ms             : uint32   4 B
    latitude                 : float    4 B  ← will be 0.0 (GPS now on Pi)
    longitude                : float    4 B  ← will be 0.0 (GPS now on Pi)
    calibration_multiplier   : float    4 B
    battery_percentage       : float    4 B
    samples[6000]            : float[] 24000 B
    end_marker               : uint32   4 B
    TOTAL                              24028 B
=============================================================================
"""

import multiprocessing as mp
import serial
import struct
import numpy as np
import time
import logging
import gc
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json

try:
    import pynmea2
    PYNMEA2_AVAILABLE = True
except ImportError:
    PYNMEA2_AVAILABLE = False
    print("[WARNING] pynmea2 not installed. GPS will be disabled.")
    print("          Install with: pip3 install pynmea2")


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Audio (ESP32-S3 via USB CDC) ─────────────────────────────────────────────
UART_PORT          = '/dev/ttyACM0'
UART_BAUD          = 115200
SAMPLE_RATE        = 48000
SAMPLES_SHORT      = 6000
PACKETS_PER_SECOND = 8

FRAME_START_MARKER = 0xAA55AA55
FRAME_END_MARKER   = 0x55AA55AA

# ── GPS (NEO-7M via GPIO UART) ────────────────────────────────────────────────
GPS_PORT               = '/dev/serial0'   # Hardware UART on Pi GPIO 14/15 '/dev/ttyAMA0'   
GPS_BAUD               = 9600
GPS_UPDATE_INTERVAL_SEC = 300             # Read GPS every 5 minutes
GPS_READ_TIMEOUT_SEC   = 30              # Max time to wait for a valid fix per session
GPS_ENABLED            = PYNMEA2_AVAILABLE  # Auto-disable if pynmea2 missing

# ── Acoustic ──────────────────────────────────────────────────────────────────
OCTAVE_FRACTION    = 3
OCTAVE_ORDER       = 6
FREQ_LIMITS        = [20, 20000]

MIC_SENSITIVITY    = -26
MIC_REF_DB         = 94.0
MIC_OFFSET_DB      = 3.0103
MIC_BITS           = 24
MIC_REF_AMPL       = pow(10, MIC_SENSITIVITY / 20) * ((1 << (MIC_BITS - 1)) - 1)

WEIGHTING_A        = 'A'
WEIGHTING_C        = 'C'

# ── Display mode ──────────────────────────────────────────────────────────────
DISPLAY_MODE       = 'second'   # 'second' or 'minute'
SECONDS_PER_MINUTE = 60

if DISPLAY_MODE not in ('second', 'minute'):
    raise ValueError(f"Invalid DISPLAY_MODE '{DISPLAY_MODE}'. Must be 'second' or 'minute'.")

# ── Frame layout ──────────────────────────────────────────────────────────────
# start_marker, timestamp_ms, lat, lon, calib, battery  →  6 fields
HEADER_FMT  = '<II4f'
HEADER_SIZE = struct.calcsize(HEADER_FMT)                        # 24 bytes
FRAME_SIZE  = HEADER_SIZE + (SAMPLES_SHORT * 4) + 4             # 24028 bytes

# ── Queue sizes ───────────────────────────────────────────────────────────────
RAW_QUEUE_SIZE    = 50
RESULT_QUEUE_SIZE = 20

# ── Plot trigger seconds ──────────────────────────────────────────────────────
PLOT_SECONDS_RAW = [10, 20]

# ── MQTT ──────────────────────────────────────────────────────────────────────
MQTT_HOST  = 'localhost'
MQTT_PORT  = 1883
MQTT_TOPIC = "kebisingan/alat1"

# =============================================================================
# HELPERS
# =============================================================================

def _resolve_plot_seconds(plot_seconds, display_mode, secs_per_min):
    if display_mode == 'second':
        return list(plot_seconds)
    snapped = [max(secs_per_min, round(s / secs_per_min) * secs_per_min)
               for s in plot_seconds]
    if snapped != list(plot_seconds):
        logging.warning(
            f"PLOT_SECONDS {plot_seconds} snapped to minute boundaries → {snapped}"
        )
    seen = set()
    return [x for x in snapped if not (x in seen or seen.add(x))]

PLOT_SECONDS = _resolve_plot_seconds(PLOT_SECONDS_RAW, DISPLAY_MODE, SECONDS_PER_MINUTE)

# ── Weighting tables ──────────────────────────────────────────────────────────
A_WEIGHTING_CORRECTIONS = np.array([
    -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4,
    -10.9,  -8.6,  -6.6,  -4.8,  -3.2,  -1.9,  -0.8,   0.0,   0.6,   1.0,
      1.2,   1.3,   1.2,   1.0,   0.5,  -0.1,  -1.1,  -2.5,  -4.3,  -6.6, -9.3
])
C_WEIGHTING_CORRECTIONS = np.array([
    -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, -0.2, -0.1,
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0
])

WEIGHTING_DICT = {
    'A': A_WEIGHTING_CORRECTIONS,
    'C': C_WEIGHTING_CORRECTIONS,
}


# =============================================================================
# SHARED GPS STATE
# =============================================================================
# mp.Array('d', 3):  [latitude, longitude, fix_valid_flag]
# mp.Value('d'):     unix timestamp of last successful fix
# Using mp.Array avoids serialization overhead of a Queue for a 3-float update.

def make_gps_state():
    """Returns (gps_array, gps_lock, gps_last_fix_time)."""
    gps_array         = mp.Array('d', [0.0, 0.0, 0.0])   # lat, lon, valid(0/1)
    gps_lock          = mp.Lock()
    gps_last_fix_time = mp.Value('d', 0.0)                # unix timestamp
    return gps_array, gps_lock, gps_last_fix_time

def gps_set(gps_array, gps_lock, gps_last_fix_time, lat, lon):
    """Thread/process-safe GPS fix update."""
    with gps_lock:
        gps_array[0] = lat
        gps_array[1] = lon
        gps_array[2] = 1.0   # valid
        gps_last_fix_time.value = time.time()

def gps_get(gps_array, gps_lock):
    """Returns (lat, lon, is_valid) — never blocks for long."""
    with gps_lock:
        return gps_array[0], gps_array[1], bool(gps_array[2])


# =============================================================================
# RING BUFFER (unchanged from original)
# =============================================================================
class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = bytearray(capacity)
        self.head     = 0
        self.tail     = 0
        self.size     = 0

    def extend(self, data):
        data_len = len(data)
        if data_len >= self.capacity:
            self.buffer[:] = data[-self.capacity:]
            self.head = 0; self.tail = 0; self.size = self.capacity
            return
        if self.tail + data_len <= self.capacity:
            self.buffer[self.tail:self.tail + data_len] = data
        else:
            first_part = self.capacity - self.tail
            self.buffer[self.tail:]              = data[:first_part]
            self.buffer[:data_len - first_part]  = data[first_part:]
        self.tail = (self.tail + data_len) % self.capacity
        self.size = min(self.size + data_len, self.capacity)

    def get_bytes(self, length):
        if length > self.size:
            return None
        if self.head + length <= self.capacity:
            return bytes(self.buffer[self.head:self.head + length])
        first_part = self.capacity - self.head
        return bytes(self.buffer[self.head:]) + bytes(self.buffer[:length - first_part])

    def consume(self, length):
        self.head = (self.head + length) % self.capacity
        self.size -= length

    def find(self, pattern):
        if self.head < self.tail:
            data = self.buffer[self.head:self.tail]
        else:
            data = bytes(self.buffer[self.head:]) + bytes(self.buffer[:self.tail])
        idx = data.find(pattern)
        return idx if idx != -1 else -1

    def __len__(self):
        return self.size


# =============================================================================
# PROCESS 0: GPS READER  (new — runs independently on its own core)
# =============================================================================
def gps_reader_process(
    gps_array:         mp.Array,
    gps_lock:          mp.Lock,
    gps_last_fix_time: mp.Value,
    stop_event:        mp.Event,
    update_interval:   int = GPS_UPDATE_INTERVAL_SEC,
    read_timeout:      int = GPS_READ_TIMEOUT_SEC,
):
    """
    Periodically opens the GPS UART, reads NMEA sentences until a valid GGA
    fix is obtained (or read_timeout expires), then closes the port and sleeps
    for update_interval seconds.

    Why open/close instead of keeping the port open?
      - Keeps the UART quiet between sessions (no NMEA flood to parse).
      - Allows other tools (gpsd, minicom) to use the port during gaps.
      - Simpler error recovery: just re-open on next cycle.

    Why GGA only?
      GGA has lat, lon, altitude, and fix quality in one sentence.
      RMC has speed and date but we don't need those here.

    GPS LED behaviour:
      The NEO-7M blinks its LED each time it sends an NMEA sentence (1Hz).
      During the read_timeout window the LED blinks continuously.
      Between sessions it is silent (port closed, no polling).
      This is expected and correct.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [GPS] %(message)s')
    logger = logging.getLogger()

    if not GPS_ENABLED:
        logger.warning("GPS disabled (pynmea2 not installed). "
                       "lat/lon will be 0.0 in all results.")
        return

    logger.info(f"GPS reader started | port={GPS_PORT} baud={GPS_BAUD} "
                f"update_interval={update_interval}s "
                f"read_timeout={read_timeout}s")

    import pynmea2   # local import — only this process needs it

    while not stop_event.is_set():
        fix_obtained = False
        session_start = time.time()

        try:
            with serial.Serial(GPS_PORT, GPS_BAUD, timeout=1) as ser:
                logger.info("GPS port open — waiting for fix...")

                while (not stop_event.is_set() and
                       not fix_obtained and
                       time.time() - session_start < read_timeout):

                    line = ser.readline().decode('ascii', errors='replace').strip()

                    if not line.startswith('$'):
                        continue

                    try:
                        msg = pynmea2.parse(line)
                    except pynmea2.ParseError:
                        continue

                    # GGA: Global Positioning System Fix Data
                    # gps_qual > 0 means valid fix (1=GPS, 2=DGPS, etc.)
                    if isinstance(msg, pynmea2.types.talker.GGA):
                        if msg.gps_qual and int(msg.gps_qual) > 0:
                            lat = float(msg.latitude)
                            lon = float(msg.longitude)
                            gps_set(gps_array, gps_lock, gps_last_fix_time,
                                    lat, lon)
                            logger.info(
                                f"Fix acquired: lat={lat:.6f} lon={lon:.6f} "
                                f"sats={msg.num_sats} alt={msg.altitude}m"
                            )
                            fix_obtained = True

        except serial.SerialException as e:
            logger.error(f"GPS serial error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue
        except Exception as e:
            logger.error(f"GPS unexpected error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        if not fix_obtained:
            logger.warning(
                f"GPS: no valid fix within {read_timeout}s. "
                f"Last known position will be used. "
                f"Retrying in {update_interval}s."
            )

        # ── Sleep until next update cycle ─────────────────────────────────
        # Use short sleeps so stop_event is checked frequently.
        sleep_end = time.time() + update_interval
        while time.time() < sleep_end and not stop_event.is_set():
            time.sleep(1)

    logger.info("GPS reader stopped.")


# =============================================================================
# PROCESS 1: SERIAL READER (audio from ESP32-S3 via USB CDC)
# =============================================================================
def serial_reader_process(
    raw_queue:          mp.Queue,
    stat_pkt_received:  mp.Value,
    stat_pkt_corrupted: mp.Value,
    stat_pkt_dropped:   mp.Value,
    stop_event:         mp.Event,
):
    """
    Unchanged from original except lat/lon from ESP32 are ignored here
    (they will be 0.0 since GPS moved to Pi). The GPS location is injected
    by the publisher at output time using the shared gps_array.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [Reader] %(message)s')
    logger = logging.getLogger()

    buffer             = RingBuffer(capacity=FRAME_SIZE * 10)
    start_marker_bytes = struct.pack('<I', FRAME_START_MARKER)

    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=0)
        logger.info(f"Audio USB opened: {UART_PORT} @ {UART_BAUD} | frame={FRAME_SIZE}B")

        while not stop_event.is_set():
            chunk = ser.read(8192)
            if chunk:
                buffer.extend(chunk)
            else:
                time.sleep(0.001)
                continue

            while len(buffer) >= FRAME_SIZE:
                # Step 1: find start marker
                first_bytes = buffer.get_bytes(4)
                if first_bytes != start_marker_bytes:
                    idx = buffer.find(start_marker_bytes)
                    if idx == -1:
                        buffer.consume(max(0, len(buffer) - 3))
                        break
                    buffer.consume(idx)
                    continue

                # Step 2: peek full frame
                frame_bytes = buffer.get_bytes(FRAME_SIZE)
                if frame_bytes is None:
                    break

                # Step 3: validate end marker
                end_marker = struct.unpack('<I', frame_bytes[-4:])[0]
                if end_marker != FRAME_END_MARKER:
                    with stat_pkt_corrupted.get_lock():
                        stat_pkt_corrupted.value += 1
                    buffer.consume(1)
                    continue

                # Step 4: parse header + samples
                try:
                    header = struct.unpack(HEADER_FMT, frame_bytes[:HEADER_SIZE])
                    # header = (start_marker, timestamp_ms, lat, lon, calib, batt)
                    # lat/lon from ESP32 ignored — GPS is now on Pi
                    calib = header[4]
                    batt  = header[5]

                    samples_end = HEADER_SIZE + SAMPLES_SHORT * 4
                    raw_samples = np.frombuffer(
                        frame_bytes[HEADER_SIZE:samples_end], dtype=np.float32
                    ).copy()

                    packet = {
                        'raw_samples': raw_samples,
                        'calib':       float(calib),
                        'batt':        float(batt),
                        # lat/lon NOT included — injected from GPS process at publish time
                    }

                    try:
                        raw_queue.put_nowait(packet)
                        with stat_pkt_received.get_lock():
                            stat_pkt_received.value += 1
                    except Exception:
                        with stat_pkt_dropped.get_lock():
                            stat_pkt_dropped.value += 1

                except Exception:
                    with stat_pkt_corrupted.get_lock():
                        stat_pkt_corrupted.value += 1

                buffer.consume(FRAME_SIZE)

    except Exception as e:
        logger.error(f"Serial Reader fatal error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        logger.info("Serial Reader stopped.")


# =============================================================================
# FILTER BANK (unchanged)
# =============================================================================
class OctaveFilterBank:
    def __init__(self, fs, fraction, order, limits):
        from octave_filter import getansifrequencies, _downsamplingfactor

        freq, freq_d, freq_u = getansifrequencies(fraction, limits)
        nyquist   = fs / 2
        valid_idx = [i for i, f in enumerate(freq_u) if f < nyquist]

        self.freq   = [freq[i]   for i in valid_idx]
        freq_d      = [freq_d[i] for i in valid_idx]
        freq_u      = [freq_u[i] for i in valid_idx]
        self.factor = _downsamplingfactor(freq_u, fs)

        self.sos_filters = []
        for lower, upper, fac in zip(freq_d, freq_u, self.factor):
            fsd = fs / fac
            sos = signal.butter(
                N=order,
                Wn=np.array([lower, upper]) / (fsd / 2),
                btype='bandpass', analog=False, output='sos',
            )
            self.sos_filters.append(sos)

        self._resample_factors = [int(round(f)) for f in self.factor]

    def filter_signal_to_leq_bands(self, x):
        leq_bands = np.zeros(len(self.freq))
        for idx, (sos, down) in enumerate(zip(self.sos_filters,
                                              self._resample_factors)):
            sd = signal.resample_poly(x, up=1, down=down) if down > 1 else x
            y  = signal.sosfilt(sos, sd)
            rms = np.sqrt(np.mean(y ** 2))
            leq_bands[idx] = (
                MIC_OFFSET_DB + MIC_REF_DB + 20 * np.log10(rms / MIC_REF_AMPL)
                if rms > 1e-10 else 0.0
            )
        return leq_bands


# =============================================================================
# LEQ AVERAGING (unchanged)
# =============================================================================
def _leq_average(leq_array_db: np.ndarray, n: int) -> np.ndarray:
    linear      = 10 ** (leq_array_db / 10)
    linear_mean = np.sum(linear, axis=0) / n
    linear_mean = np.where(linear_mean > 1e-10, linear_mean, 1e-10)
    return 10 * np.log10(linear_mean)


# =============================================================================
# PROCESS 2: OCTAVE ANALYZER
# =============================================================================
def octave_leq_analyzer_process(
    raw_queue:    mp.Queue,
    result_queue: mp.Queue,
    weighting_a:  str,
    weighting_c:  str,
    display_mode: str,
    stat_second:  mp.Value,
    stop_event:   mp.Event,
):
    """
    Unchanged from original. lat/lon removed — GPS is injected by publisher.
    batt is still passed through result dict.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [Analyzer] %(message)s')
    logger = logging.getLogger()
    logger.info(f"Analyzer started | All weighting | mode={display_mode}")

    filter_bank             = OctaveFilterBank(SAMPLE_RATE, OCTAVE_FRACTION,
                                               OCTAVE_ORDER, FREQ_LIMITS)
    weighting_corrections_a = WEIGHTING_DICT[weighting_a]
    weighting_corrections_c = WEIGHTING_DICT[weighting_c]
    n_bands                 = len(filter_bank.freq)

    packet_counter = 0
    all_samples    = np.zeros(SAMPLES_SHORT * PACKETS_PER_SECOND, dtype=np.float32)
    second_count   = 1
    current_batt   = 0.0

    minute_buffer_bands            = np.zeros((SECONDS_PER_MINUTE, n_bands), dtype=np.float64)
    minute_buffer_total_raw        = np.zeros(SECONDS_PER_MINUTE, dtype=np.float64)
    minute_buffer_total_weighted_a = np.zeros(SECONDS_PER_MINUTE, dtype=np.float64)
    minute_buffer_total_weighted_c = np.zeros(SECONDS_PER_MINUTE, dtype=np.float64)
    minute_second_idx              = 0
    last_batt                      = 0.0

    while not stop_event.is_set():
        try:
            packet = raw_queue.get(timeout=1.0)
        except Exception:
            continue

        raw_samples  = packet['raw_samples']
        calib        = packet['calib']
        samples      = raw_samples * calib
        current_batt = packet['batt']
        # Add this temporarily to the analyzer, right after samples = raw_samples * calib:
        print(f"calib={calib} rms={np.sqrt(np.mean(samples**2)):.1f} max={np.max(np.abs(samples)):.1f}")

        start_idx = packet_counter * SAMPLES_SHORT
        all_samples[start_idx:start_idx + SAMPLES_SHORT] = samples
        packet_counter += 1

        if packet_counter < PACKETS_PER_SECOND:
            continue

        # ── 1-second Leq ─────────────────────────────────────────────────
        leq_bands_raw        = filter_bank.filter_signal_to_leq_bands(all_samples)
        leq_bands_weighted_a = leq_bands_raw + weighting_corrections_a
        leq_bands_weighted_c = leq_bands_raw + weighting_corrections_c

        linear_sum_w_a       = np.sum(10 ** (leq_bands_weighted_a / 10))
        leq_total_weighted_a = 10 * np.log10(linear_sum_w_a) if linear_sum_w_a > 1e-10 else 0.0

        linear_sum_w_c       = np.sum(10 ** (leq_bands_weighted_c / 10))
        leq_total_weighted_c = 10 * np.log10(linear_sum_w_c) if linear_sum_w_c > 1e-10 else 0.0

        linear_sum_r  = np.sum(10 ** (leq_bands_raw / 10))
        leq_total_raw = 10 * np.log10(linear_sum_r) if linear_sum_r > 1e-10 else 0.0

        timestamp = time.strftime("%H:%M:%S")
        last_batt = current_batt

        if display_mode == 'second':
            result = {
                'leq_total_weighted_a': float(leq_total_weighted_a),
                'leq_total_weighted_c': float(leq_total_weighted_c),
                'leq_total_raw':        float(leq_total_raw),
                'leq_bands_raw':        leq_bands_raw.tolist(),
                'freq_bands':           filter_bank.freq,
                'display_mode':         'second',
                'timestamp':            timestamp,
                'batt':                 float(current_batt),
                'second':               second_count,
                'label':                f"Second #{second_count}",
                'gain':                 calib,
                # lat/lon intentionally absent — publisher injects from GPS process
            }
            try:
                result_queue.put_nowait(result)
            except Exception:
                pass

        elif display_mode == 'minute':
            minute_buffer_bands[minute_second_idx]            = leq_bands_raw
            minute_buffer_total_raw[minute_second_idx]        = leq_total_raw
            minute_buffer_total_weighted_a[minute_second_idx] = leq_total_weighted_a
            minute_buffer_total_weighted_c[minute_second_idx] = leq_total_weighted_c
            minute_second_idx += 1

            if minute_second_idx >= SECONDS_PER_MINUTE:
                minute_count = second_count // SECONDS_PER_MINUTE

                leq_1min_bands_raw = _leq_average(minute_buffer_bands, SECONDS_PER_MINUTE)
                leq_1min_total_raw = float(_leq_average(
                    minute_buffer_total_raw, SECONDS_PER_MINUTE))
                leq_1min_total_weighted_a = float(_leq_average(
                    minute_buffer_total_weighted_a, SECONDS_PER_MINUTE))
                leq_1min_total_weighted_c = float(_leq_average(
                    minute_buffer_total_weighted_c, SECONDS_PER_MINUTE))

                result = {
                    'leq_total_weighted_a': leq_1min_total_weighted_a,
                    'leq_total_weighted_c': leq_1min_total_weighted_c,
                    'leq_total_raw':        leq_1min_total_raw,
                    'leq_bands_raw':        leq_1min_bands_raw.tolist(),
                    'freq_bands':           filter_bank.freq,
                    'display_mode':         'minute',
                    'timestamp':            timestamp,
                    'batt':                 float(last_batt),
                    'second':               second_count,
                    'minute':               minute_count,
                    'label':                f"Minute #{minute_count} (second #{second_count})",
                    'gain':                 calib,
                }
                try:
                    result_queue.put_nowait(result)
                except Exception:
                    pass

                minute_second_idx = 0
                minute_buffer_bands[:]            = 0.0
                minute_buffer_total_raw[:]        = 0.0
                minute_buffer_total_weighted_a[:] = 0.0
                minute_buffer_total_weighted_c[:] = 0.0

        packet_counter = 0
        second_count  += 1
        with stat_second.get_lock():
            stat_second.value = second_count

    logger.info("Analyzer stopped.")


# =============================================================================
# PROCESS 3: PUBLISHER
# =============================================================================
def publisher_process(
    result_queue:      mp.Queue,
    gps_array:         mp.Array,
    gps_lock:          mp.Lock,
    gps_last_fix_time: mp.Value,
    plot_seconds:      list,
    stop_event:        mp.Event,
):
    """
    GPS INJECTION HAPPENS HERE.

    At publish time, the publisher reads the latest GPS fix from the shared
    mp.Array. This is the correct place to inject location data because:

      1. No synchronization overhead — one lock acquisition per result,
         not per audio packet (8x per second).
      2. The GPS fix is valid for the entire publish interval (5 minutes).
         There is no meaningful "which packet does this fix belong to" question.
      3. The publisher is the last stage — it owns the final output record
         and is responsible for merging all data sources into it.

    If no valid fix has been received, lat/lon are 0.0 and a warning is logged.
    The 'gps_valid' field in the MQTT payload indicates fix status.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [Publisher] %(message)s')
    logger = logging.getLogger()
    logger.info(f"Publisher started | plot at seconds: {plot_seconds}")

    try:
        client = mqtt.Client()
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.loop_start()
        logger.info("Connected to MQTT")
    except Exception as e:
        logger.error(f"MQTT connect failed: {e}")
        client = None

    plotted_seconds  = set()
    plot_seconds_set = set(plot_seconds)

    nominal_labels = [
        "20", "25", "31.5", "40", "50", "63", "80", "100", "125", "160",
        "200", "250", "315", "400", "500", "630", "800", "1k", "1.25k",
        "1.6k", "2k", "2.5k", "3.15k", "4k", "5k", "6.3k", "8k",
        "10k", "12.5k", "16k", "20k",
    ]

    freq_keys = [
        "freq_00020", "freq_00025", "freq_00031_5", "freq_00040",
        "freq_00050", "freq_00063", "freq_00080", "freq_00100",
        "freq_00125", "freq_00160", "freq_00200", "freq_00250",
        "freq_00315", "freq_00400", "freq_00500", "freq_00630",
        "freq_00800", "freq_01000", "freq_01250", "freq_01600",
        "freq_02000", "freq_02500", "freq_03150", "freq_04000",
        "freq_05000", "freq_06300", "freq_08000", "freq_10000",
        "freq_12500", "freq_16000", "freq_20000",
    ]

    while not stop_event.is_set():
        try:
            result = result_queue.get(timeout=1.0)
        except Exception:
            continue

        # ── Inject GPS location ───────────────────────────────────────────
        # Read the latest fix from the shared array. This is always
        # the most recent position the GPS process has written.
        lat, lon, gps_valid = gps_get(gps_array, gps_lock)

        if not gps_valid:
            # Log a warning only occasionally to avoid log spam
            if int(time.time()) % 30 == 0:
                logger.warning(
                    "No valid GPS fix yet — publishing with lat=0.0 lon=0.0. "
                    "Check antenna and sky view. "
                    f"GPS update interval: {GPS_UPDATE_INTERVAL_SEC}s."
                )

        # ── Log ───────────────────────────────────────────────────────────
        ts          = result['timestamp']
        leq_z       = result['leq_total_raw']
        leq_a       = result['leq_total_weighted_a']
        leq_c       = result['leq_total_weighted_c']
        label       = result['label']
        mode        = result['display_mode']
        batt        = result.get('batt', 0.0)
        calibration = result['gain']
        unit        = "minute" if mode == 'minute' else "second"
        fix_str     = f"lat={lat:.6f} lon={lon:.6f}" if gps_valid else "no GPS fix"

        logger.info(
            f"[{ts}] {label}: "
            f"LZeq={leq_z:.1f}dBZ  LAeq={leq_a:.1f}dBA  LCeq={leq_c:.1f}dBC  "
            f"({unit})  batt={batt:.1f}%  gain={calibration:.3f}  {fix_str}"
        )

        # ── Plot ──────────────────────────────────────────────────────────
        sec = result['second']
        if sec in plot_seconds_set and sec not in plotted_seconds:
            _plot_result(result, nominal_labels, logger)
            plotted_seconds.add(sec)

        # ── MQTT payload ──────────────────────────────────────────────────
        payload = {
            'no_dev':               'dev_001',
            'timestamp':            ts,
            'leq_total_weighted_a': leq_a,
            'leq_total_weighted_c': leq_c,
            'leq_total':            leq_z,
            'lat':                  lat,        # from GPS process
            'lon':                  lon,        # from GPS process
            'gps_valid':            gps_valid,  # consumer can check fix quality
            'batt':                 batt,
            'gain':                 calibration,
        }
        for i, key in enumerate(freq_keys):
            payload[key] = result['leq_bands_raw'][i]

        if client:
            try:
                client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
            except Exception as e:
                logger.error(f"MQTT publish error: {e}")

    if client:
        client.loop_stop()
        client.disconnect()
        logger.info("MQTT disconnected.")

    logger.info("Publisher stopped.")


# =============================================================================
# PLOT HELPER (unchanged)
# =============================================================================
def _plot_result(result, nominal_labels, logger):
    try:
        leq_bands_raw        = result['leq_bands_raw']
        leq_total_raw        = result['leq_total_raw']
        leq_total_weighted_a = result['leq_total_weighted_a']
        leq_total_weighted_c = result['leq_total_weighted_c']
        freq_bands           = result['freq_bands']
        label                = result['label']
        mode                 = result['display_mode']

        freq_labels = nominal_labels[:len(freq_bands)]
        fig, ax     = plt.subplots(figsize=(14, 7))
        x_pos       = np.arange(len(freq_bands))
        bars        = ax.bar(x_pos, leq_bands_raw, color='steelblue',
                             alpha=0.7, edgecolor='black', linewidth=0.5)

        for bar, level in zip(bars, leq_bands_raw):
            bar.set_color('red' if level > 80 else ('orange' if level > 65 else 'green'))
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                    f'{level:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('LZeq (dBZ)', fontsize=12, fontweight='bold')

        duration_str = "1-minute" if mode == 'minute' else "1-second"
        ax.set_title(
            f'1/3 Octave Band LZeq [{duration_str}] [{label}]\n'
            f'LZeq={leq_total_raw:.1f}dBZ  '
            f'LAeq={leq_total_weighted_a:.1f}dBA  '
            f'LCeq={leq_total_weighted_c:.1f}dBC',
            fontsize=14, fontweight='bold'
        )
        ax.set_ylim(20, 125)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()

        filename = f"leq_{mode}_s{result['second']}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {filename}")
        plt.close(fig)
        gc.collect()

    except Exception as e:
        logger.error(f"Plot error: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    mp.set_start_method('spawn')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Integrated Noise Monitoring System")
    logger.info(f"Audio source : {UART_PORT} (ESP32-S3 USB CDC)")
    logger.info(f"GPS source   : {GPS_PORT} (NEO-7M GPIO UART)"
                if GPS_ENABLED else "GPS source   : DISABLED (pynmea2 missing)")
    logger.info(f"GPS interval : every {GPS_UPDATE_INTERVAL_SEC}s "
                f"(~{GPS_UPDATE_INTERVAL_SEC // 60} min)")
    logger.info(f"GPS timeout  : {GPS_READ_TIMEOUT_SEC}s per session")
    logger.info(f"Weighting    : A + C + Z (all)")
    logger.info(f"Display Mode : {DISPLAY_MODE}")
    logger.info(f"CPU cores    : {mp.cpu_count()}")
    logger.info(f"Frame size   : {FRAME_SIZE} bytes")
    logger.info("=" * 70)

    # ── Shared GPS state (cross-process via mp.Array) ─────────────────────
    gps_array, gps_lock, gps_last_fix_time = make_gps_state()

    # ── Shared queues ─────────────────────────────────────────────────────
    raw_queue    = mp.Queue(maxsize=RAW_QUEUE_SIZE)
    result_queue = mp.Queue(maxsize=RESULT_QUEUE_SIZE)
    stop_event   = mp.Event()

    stat_pkt_received  = mp.Value('i', 0)
    stat_pkt_corrupted = mp.Value('i', 0)
    stat_pkt_dropped   = mp.Value('i', 0)
    stat_second        = mp.Value('i', 0)

    # ── Processes ─────────────────────────────────────────────────────────
    p_gps = mp.Process(
        target=gps_reader_process,
        args=(gps_array, gps_lock, gps_last_fix_time, stop_event,
              GPS_UPDATE_INTERVAL_SEC, GPS_READ_TIMEOUT_SEC),
        name="GpsReader",
        daemon=True,
    )
    p_reader = mp.Process(
        target=serial_reader_process,
        args=(raw_queue, stat_pkt_received, stat_pkt_corrupted,
              stat_pkt_dropped, stop_event),
        name="SerialReader",
        daemon=True,
    )
    p_analyzer = mp.Process(
        target=octave_leq_analyzer_process,
        args=(raw_queue, result_queue, WEIGHTING_A, WEIGHTING_C,
              DISPLAY_MODE, stat_second, stop_event),
        name="OctaveAnalyzer",
        daemon=True,
    )
    p_publisher = mp.Process(
        target=publisher_process,
        args=(result_queue, gps_array, gps_lock, gps_last_fix_time,
              PLOT_SECONDS, stop_event),
        name="Publisher",
        daemon=True,
    )

    # GPS starts first — give it a head start acquiring a fix
    p_gps.start()
    time.sleep(0.5)

    p_reader.start()
    time.sleep(0.3)
    p_analyzer.start()
    time.sleep(0.2)
    p_publisher.start()

    logger.info("All processes started. Press Ctrl+C to stop.")
    logger.info("=" * 70)

    try:
        while True:
            time.sleep(1)
            if not p_reader.is_alive():
                logger.error("SerialReader died unexpectedly!")
                break
            if not p_analyzer.is_alive():
                logger.error("OctaveAnalyzer died unexpectedly!")
                break
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        logger.info("Stopping all processes...")
    finally:
        stop_event.set()

        for p in [p_gps, p_reader, p_analyzer, p_publisher]:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"{p.name} did not stop cleanly — terminating.")
                p.terminate()

        print("\n" + "=" * 70)
        logger.info("=== Final Statistics ===")
        logger.info(f"Packets received  : {stat_pkt_received.value}")
        logger.info(f"Packets corrupted : {stat_pkt_corrupted.value}")
        logger.info(f"Packets dropped   : {stat_pkt_dropped.value}")
        total = stat_pkt_received.value + stat_pkt_corrupted.value + stat_pkt_dropped.value
        if total > 0:
            logger.info(f"Success rate      : {stat_pkt_received.value / total * 100:.2f}%")
        logger.info(f"Total seconds     : {stat_second.value}")
        if DISPLAY_MODE == 'minute':
            logger.info(f"Total minutes     : {stat_second.value // SECONDS_PER_MINUTE}")

        lat, lon, valid = gps_get(gps_array, gps_lock)
        if valid:
            logger.info(f"Last GPS fix      : lat={lat:.6f} lon={lon:.6f}")
        else:
            logger.info("Last GPS fix      : no valid fix received")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()