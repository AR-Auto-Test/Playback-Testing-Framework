package com.google.mediapipe.components;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.os.Build;
import android.os.Process;
import android.util.Log;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.ByteBuffer;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/MicrophoneHelper.class */
public class MicrophoneHelper implements AudioDataProducer {
    private static final String TAG = "MicrophoneHelper";
    private static final int AUDIO_ENCODING = 2;
    private static final int AUDIO_SOURCE = 1;
    private static final int BUFFER_SIZE_MULTIPLIER = 2;
    private static final long DEFAULT_READ_INTERVAL_MICROS = 10000;
    private static final int BYTES_PER_SAMPLE = 2;
    private static final long UNINITIALIZED_TIMESTAMP = Long.MIN_VALUE;
    private static final long NANOS_PER_MICROS = 1000;
    private static final long MICROS_PER_SECOND = 1000000;
    private static final long NANOS_PER_SECOND = 1000000000;
    private final int sampleRateInHz;
    private final int channelConfig;
    private final int bytesPerFrame;
    private final int minBufferSize;
    private int audioRecordBufferSize;
    private int audioPacketBufferSize;
    private AudioRecord audioRecord;
    private AudioFormat audioFormat;
    private Thread recordingThread;
    private AudioDataConsumer consumer;
    private long readIntervalMicros = 10000;
    private long initialTimestampNanos = Long.MIN_VALUE;
    private long startRecordingTimestampNanos = Long.MIN_VALUE;
    private boolean recording = false;

    public MicrophoneHelper(int sampleRateInHz, int channelConfig) {
        this.sampleRateInHz = sampleRateInHz;
        this.channelConfig = channelConfig;
        int numChannels = channelConfig == 12 ? 2 : 1;
        this.bytesPerFrame = 2 * numChannels;
        this.minBufferSize = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, 2);
        updateBufferSizes(this.readIntervalMicros);
    }

    public void setReadIntervalMicros(long micros) {
        this.readIntervalMicros = micros;
        updateBufferSizes(this.readIntervalMicros);
    }

    private void updateBufferSizes(long micros) {
        this.audioPacketBufferSize = (int) Math.ceil((((1.0d * this.bytesPerFrame) * this.sampleRateInHz) * micros) / 1000000.0d);
        this.audioRecordBufferSize = Math.max(this.audioPacketBufferSize, this.minBufferSize) * 2;
    }

    private void setupAudioRecord() {
        Log.d("MicrophoneHelper", "AudioRecord(" + this.sampleRateInHz + ", " + this.audioRecordBufferSize + ")");
        this.audioFormat = new AudioFormat.Builder().setEncoding(2).setSampleRate(this.sampleRateInHz).setChannelMask(this.channelConfig).build();
        this.audioRecord = new AudioRecord.Builder().setAudioSource(1).setAudioFormat(this.audioFormat).setBufferSizeInBytes(this.audioRecordBufferSize).build();
        if (this.audioRecord.getState() != 1) {
            this.audioRecord.release();
            Log.e("MicrophoneHelper", "AudioRecord could not open.");
            return;
        }
        this.recordingThread = new Thread(() -> {
            Process.setThreadPriority(-16);
            this.startRecordingTimestampNanos = System.nanoTime();
            long timestampOffsetNanos = 0;
            int totalNumFramesRead = 0;
            while (this.recording && this.audioRecord != null) {
                ByteBuffer audioData = ByteBuffer.allocateDirect(this.audioPacketBufferSize);
                try {
                    readAudioPacket(audioData);
                    long timestampNanos = getTimestampNanos(totalNumFramesRead);
                    if (totalNumFramesRead == 0 && this.initialTimestampNanos != Long.MIN_VALUE) {
                        timestampOffsetNanos = timestampNanos - this.initialTimestampNanos;
                    }
                    long timestampMicros = (timestampNanos - timestampOffsetNanos) / 1000;
                    int numFramesRead = audioData.limit() / this.bytesPerFrame;
                    totalNumFramesRead += numFramesRead;
                    if (this.recording && this.consumer != null) {
                        this.consumer.onNewAudioData(audioData, timestampMicros, this.audioFormat);
                    }
                } catch (IOException ioException) {
                    Log.e("MicrophoneHelper", ioException.getMessage());
                }
            }
        }, "microphoneHelperRecordingThread");
    }

    private void readAudioPacket(ByteBuffer audioPacket) throws IOException {
        int numBytesRead;
        int totalNumBytesRead = 0;
        while (totalNumBytesRead < audioPacket.capacity()) {
            int bytesRemaining = audioPacket.capacity() - totalNumBytesRead;
            if (Build.VERSION.SDK_INT >= 23) {
                numBytesRead = this.audioRecord.read(audioPacket, bytesRemaining, 0);
            } else {
                numBytesRead = this.audioRecord.read(audioPacket, bytesRemaining);
            }
            if (numBytesRead <= 0) {
                String error = "ERROR";
                if (numBytesRead == -3) {
                    error = "ERROR_INVALID_OPERATION";
                } else if (numBytesRead == -2) {
                    error = "ERROR_BAD_VALUE";
                } else if (numBytesRead == -6) {
                    error = "ERROR_DEAD_OBJECT";
                }
                throw new IOException("AudioRecord.read(...) failed due to " + error);
            }
            totalNumBytesRead += numBytesRead;
            audioPacket.position(totalNumBytesRead);
        }
        audioPacket.position(0);
    }

    private long getTimestampNanos(long framePosition) {
        long referenceFrame = 0;
        long referenceTimestamp = this.startRecordingTimestampNanos;
        AudioTimestamp audioTimestamp = getAudioRecordTimestamp();
        if (audioTimestamp != null) {
            referenceFrame = audioTimestamp.framePosition;
            referenceTimestamp = audioTimestamp.nanoTime;
        }
        return referenceTimestamp + (((framePosition - referenceFrame) * 1000000000) / this.sampleRateInHz);
    }

    private AudioTimestamp getAudioRecordTimestamp() {
        Preconditions.checkNotNull(this.audioRecord);
        if (Build.VERSION.SDK_INT >= 24) {
            AudioTimestamp audioTimestamp = new AudioTimestamp();
            int status = this.audioRecord.getTimestamp(audioTimestamp, 0);
            if (status == 0) {
                return audioTimestamp;
            }
            Log.e("MicrophoneHelper", "audioRecord.getTimestamp failed with status: " + status);
            return null;
        }
        return null;
    }

    public int getAudioRecordBufferSize() {
        return this.audioRecordBufferSize;
    }

    public int getAudioPacketBufferSize() {
        return this.audioPacketBufferSize;
    }

    public void setInitialTimestampNanos(long initialTimestampNanos) {
        this.initialTimestampNanos = initialTimestampNanos;
    }

    public void startMicrophone() {
        if (this.recording) {
            return;
        }
        setupAudioRecord();
        this.audioRecord.startRecording();
        if (this.audioRecord.getRecordingState() != 3) {
            Log.e("MicrophoneHelper", "AudioRecord couldn't start recording.");
            this.audioRecord.release();
            return;
        }
        this.recording = true;
        this.recordingThread.start();
        Log.d("MicrophoneHelper", "AudioRecord is recording audio.");
    }

    public void stopMicrophone() {
        stopMicrophoneWithoutCleanup();
        cleanup();
        Log.d("MicrophoneHelper", "AudioRecord stopped recording audio.");
    }

    public void stopMicrophoneWithoutCleanup() {
        Preconditions.checkNotNull(this.audioRecord);
        if (!this.recording) {
            return;
        }
        this.recording = false;
        try {
            if (this.recordingThread != null) {
                this.recordingThread.join();
            }
        } catch (InterruptedException ie) {
            Log.e("MicrophoneHelper", "Exception: ", ie);
        }
        this.audioRecord.stop();
        if (this.audioRecord.getRecordingState() != 1) {
            Log.e("MicrophoneHelper", "AudioRecord.stop() didn't run properly.");
        }
    }

    public void cleanup() {
        Preconditions.checkNotNull(this.audioRecord);
        if (this.recording) {
            return;
        }
        this.audioRecord.release();
    }

    public void setAudioConsumer(AudioDataConsumer consumer) {
        this.consumer = consumer;
    }
}