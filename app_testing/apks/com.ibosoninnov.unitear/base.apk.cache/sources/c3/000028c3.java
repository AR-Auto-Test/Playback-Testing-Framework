package com.google.mediapipe.components;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.AudioFormat;
import android.os.Handler;
import android.util.Log;
import com.google.common.base.Preconditions;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Graph;
import com.google.mediapipe.framework.GraphService;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketCallback;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.SurfaceOutput;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.proto.CalculatorProto;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/FrameProcessor.class */
public class FrameProcessor implements TextureFrameProcessor, AudioDataProcessor {
    private static final String TAG = "FrameProcessor";
    private static final int BYTES_PER_MONO_SAMPLE = 2;
    private static final int AUDIO_ENCODING = 2;
    private Graph mediapipeGraph;
    private AndroidPacketCreator packetCreator;
    private OnWillAddFrameListener addFrameListener;
    private ErrorListener asyncErrorListener;
    private String videoInputStream;
    private String videoInputStreamCpu;
    private String videoOutputStream;
    private SurfaceOutput videoSurfaceOutput;
    private String audioInputStream;
    private String audioOutputStream;
    private double audioSampleRate;
    private List<TextureFrameConsumer> videoConsumers = new ArrayList();
    private List<AudioDataConsumer> audioConsumers = new ArrayList();
    private final AtomicBoolean started = new AtomicBoolean(false);
    private int numAudioChannels = 1;

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/FrameProcessor$ErrorListener.class */
    public interface ErrorListener {
        void onError(RuntimeException error);
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/FrameProcessor$OnWillAddFrameListener.class */
    public interface OnWillAddFrameListener {
        void onWillAddFrame(long timestamp);
    }

    public FrameProcessor(Context context, long parentNativeContext, String graphName, String inputStream, @Nullable String outputStream) {
        try {
            initializeGraphAndPacketCreator(context, graphName);
            addVideoStreams(parentNativeContext, inputStream, outputStream);
        } catch (MediaPipeException e2) {
            Log.e("FrameProcessor", "MediaPipe error: ", e2);
        }
    }

    public FrameProcessor(Context context, String graphName) {
        initializeGraphAndPacketCreator(context, graphName);
    }

    public FrameProcessor(CalculatorProto.CalculatorGraphConfig graphConfig) {
        initializeGraphAndPacketCreator(graphConfig);
    }

    private void initializeGraphAndPacketCreator(Context context, String graphName) {
        this.mediapipeGraph = new Graph();
        if (new File(graphName).isAbsolute()) {
            this.mediapipeGraph.loadBinaryGraph(graphName);
        } else {
            this.mediapipeGraph.loadBinaryGraph(AndroidAssetUtil.getAssetBytes(context.getAssets(), graphName));
        }
        this.packetCreator = new AndroidPacketCreator(this.mediapipeGraph);
    }

    private void initializeGraphAndPacketCreator(CalculatorProto.CalculatorGraphConfig graphConfig) {
        this.mediapipeGraph = new Graph();
        this.mediapipeGraph.loadBinaryGraph(graphConfig);
        this.packetCreator = new AndroidPacketCreator(this.mediapipeGraph);
    }

    public void setAsynchronousErrorListener(@Nullable ErrorListener listener) {
        this.asyncErrorListener = listener;
    }

    public void setAsynchronousErrorListener(@Nullable ErrorListener listener, @Nullable Handler handler) {
        ErrorListener errorListener;
        if (handler == null) {
            errorListener = listener;
        } else {
            errorListener = e2 -> {
                handler.post(() -> {
                    listener.onError(e2);
                });
            };
        }
        setAsynchronousErrorListener(errorListener);
    }

    public void addVideoStreams(long parentNativeContext, @Nullable String inputStream, @Nullable String outputStream) {
        this.videoInputStream = inputStream;
        this.videoOutputStream = outputStream;
        this.mediapipeGraph.setParentGlContext(parentNativeContext);
        if (this.videoOutputStream != null) {
            this.mediapipeGraph.addPacketCallback(this.videoOutputStream, new PacketCallback() { // from class: com.google.mediapipe.components.FrameProcessor.1
                @Override // com.google.mediapipe.framework.PacketCallback
                public void process(Packet packet) {
                    List<TextureFrameConsumer> currentConsumers;
                    synchronized (this) {
                        currentConsumers = FrameProcessor.this.videoConsumers;
                    }
                    for (TextureFrameConsumer consumer : currentConsumers) {
                        TextureFrame frame = PacketGetter.getTextureFrame(packet);
                        if (Log.isLoggable("FrameProcessor", 2)) {
                            Log.v("FrameProcessor", String.format("Output tex: %d width: %d height: %d to consumer %h", Integer.valueOf(frame.getTextureName()), Integer.valueOf(frame.getWidth()), Integer.valueOf(frame.getHeight()), consumer));
                        }
                        consumer.onNewFrame(frame);
                    }
                }
            });
            this.videoSurfaceOutput = this.mediapipeGraph.addSurfaceOutput(this.videoOutputStream);
        }
    }

    public void addAudioStreams(@Nullable String inputStream, @Nullable String outputStream, int numInputChannels, int numOutputChannels, double audioSampleRateInHz) {
        this.audioInputStream = inputStream;
        this.audioOutputStream = outputStream;
        this.numAudioChannels = numInputChannels;
        this.audioSampleRate = audioSampleRateInHz;
        if (this.audioInputStream != null) {
            Packet audioHeader = this.packetCreator.createTimeSeriesHeader(this.numAudioChannels, this.audioSampleRate);
            this.mediapipeGraph.setStreamHeader(this.audioInputStream, audioHeader);
        }
        if (this.audioOutputStream != null) {
            int outputAudioChannelMask = numOutputChannels == 2 ? 12 : 16;
            final AudioFormat audioFormat = new AudioFormat.Builder().setEncoding(2).setSampleRate((int) this.audioSampleRate).setChannelMask(outputAudioChannelMask).build();
            this.mediapipeGraph.addPacketCallback(this.audioOutputStream, new PacketCallback() { // from class: com.google.mediapipe.components.FrameProcessor.2
                @Override // com.google.mediapipe.framework.PacketCallback
                public void process(Packet packet) {
                    List<AudioDataConsumer> currentAudioConsumers;
                    synchronized (this) {
                        currentAudioConsumers = FrameProcessor.this.audioConsumers;
                    }
                    for (AudioDataConsumer consumer : currentAudioConsumers) {
                        byte[] buffer = PacketGetter.getAudioByteData(packet);
                        ByteBuffer audioData = ByteBuffer.wrap(buffer);
                        consumer.onNewAudioData(audioData, packet.getTimestamp(), audioFormat);
                    }
                }
            });
        }
    }

    public synchronized <T> void setServiceObject(GraphService<T> service, T object) {
        this.mediapipeGraph.setServiceObject(service, object);
    }

    public void setInputSidePackets(Map<String, Packet> inputSidePackets) {
        Preconditions.checkState(!this.started.get(), "setInputSidePackets must be called before the graph is started");
        this.mediapipeGraph.setInputSidePackets(inputSidePackets);
    }

    @Override // com.google.mediapipe.components.TextureFrameProducer
    public void setConsumer(TextureFrameConsumer consumer) {
        synchronized (this) {
            this.videoConsumers = Arrays.asList(consumer);
        }
    }

    public void setAudioConsumer(AudioDataConsumer consumer) {
        synchronized (this) {
            this.audioConsumers = Arrays.asList(consumer);
        }
    }

    public void setVideoInputStreamCpu(String inputStream) {
        this.videoInputStreamCpu = inputStream;
    }

    public void addPacketCallback(String outputStream, PacketCallback callback) {
        this.mediapipeGraph.addPacketCallback(outputStream, callback);
    }

    public void addConsumer(TextureFrameConsumer consumer) {
        synchronized (this) {
            List<TextureFrameConsumer> newConsumers = new ArrayList<>(this.videoConsumers);
            newConsumers.add(consumer);
            this.videoConsumers = newConsumers;
        }
    }

    public boolean removeConsumer(TextureFrameConsumer listener) {
        boolean existed;
        synchronized (this) {
            List<TextureFrameConsumer> newConsumers = new ArrayList<>(this.videoConsumers);
            existed = newConsumers.remove(listener);
            this.videoConsumers = newConsumers;
        }
        return existed;
    }

    public Graph getGraph() {
        return this.mediapipeGraph;
    }

    public AndroidPacketCreator getPacketCreator() {
        return this.packetCreator;
    }

    public SurfaceOutput getVideoSurfaceOutput() {
        return this.videoSurfaceOutput;
    }

    public void close() {
        if (this.started.get()) {
            try {
                this.mediapipeGraph.closeAllPacketSources();
                this.mediapipeGraph.waitUntilGraphDone();
            } catch (MediaPipeException e2) {
                if (this.asyncErrorListener != null) {
                    this.asyncErrorListener.onError(e2);
                } else {
                    Log.e("FrameProcessor", "Mediapipe error: ", e2);
                }
            }
            try {
                this.mediapipeGraph.tearDown();
            } catch (MediaPipeException e3) {
                Log.e("FrameProcessor", "Mediapipe error: ", e3);
            }
        }
    }

    public void preheat() {
        if (!this.started.getAndSet(true)) {
            startGraph();
        }
    }

    public void setOnWillAddFrameListener(@Nullable OnWillAddFrameListener addFrameListener) {
        this.addFrameListener = addFrameListener;
    }

    private boolean maybeAcceptNewFrame(long timestamp) {
        if (!this.started.getAndSet(true)) {
            startGraph();
            return true;
        }
        return true;
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[]}, finally: {[INVOKE, IF, INVOKE, INVOKE, IF, IF] complete} */
    /* JADX DEBUG: Don't trust debug lines info. Repeating lines: [461=4, 464=4, 466=4, 469=4] */
    @Override // com.google.mediapipe.components.TextureFrameConsumer
    public void onNewFrame(TextureFrame frame) {
        Packet imagePacket = null;
        long timestamp = frame.getTimestamp();
        try {
            try {
                if (Log.isLoggable("FrameProcessor", 2)) {
                    Log.v("FrameProcessor", String.format("Input tex: %d width: %d height: %d", Integer.valueOf(frame.getTextureName()), Integer.valueOf(frame.getWidth()), Integer.valueOf(frame.getHeight())));
                }
                if (!maybeAcceptNewFrame(frame.getTimestamp())) {
                    if (0 != 0) {
                        imagePacket.release();
                    }
                    if (frame != null) {
                        frame.release();
                        return;
                    }
                    return;
                }
                if (this.addFrameListener != null) {
                    this.addFrameListener.onWillAddFrame(timestamp);
                }
                Packet imagePacket2 = this.packetCreator.createGpuBuffer(frame);
                TextureFrame frame2 = null;
                try {
                    this.mediapipeGraph.addConsumablePacketToInputStream(this.videoInputStream, imagePacket2, timestamp);
                    imagePacket2 = null;
                } catch (MediaPipeException e2) {
                    if (this.asyncErrorListener != null) {
                        throw e2;
                    }
                    Log.e("FrameProcessor", "Mediapipe error: ", e2);
                }
                if (imagePacket2 != null) {
                    imagePacket2.release();
                }
                if (0 != 0) {
                    frame2.release();
                }
            } catch (RuntimeException e3) {
                if (this.asyncErrorListener == null) {
                    throw e3;
                }
                this.asyncErrorListener.onError(e3);
                if (0 != 0) {
                    imagePacket.release();
                }
                if (frame != null) {
                    frame.release();
                }
            }
        } catch (Throwable th) {
            if (0 != 0) {
                imagePacket.release();
            }
            if (frame != null) {
                frame.release();
            }
            throw th;
        }
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[]}, finally: {[INVOKE, IF] complete} */
    /* JADX DEBUG: Don't trust debug lines info. Repeating lines: [513=4, 514=4] */
    public void onNewFrame(final Bitmap bitmap, long timestamp) {
        Packet packet = null;
        try {
            try {
                if (!maybeAcceptNewFrame(timestamp)) {
                    if (0 != 0) {
                        packet.release();
                        return;
                    }
                    return;
                }
                if (this.addFrameListener != null) {
                    this.addFrameListener.onWillAddFrame(timestamp);
                }
                Packet packet2 = getPacketCreator().createRgbImageFrame(bitmap);
                try {
                    this.mediapipeGraph.addConsumablePacketToInputStream(this.videoInputStreamCpu, packet2, timestamp);
                    packet2 = null;
                } catch (MediaPipeException e2) {
                    if (this.asyncErrorListener != null) {
                        throw e2;
                    }
                    Log.e("FrameProcessor", "Mediapipe error: ", e2);
                }
                if (packet2 != null) {
                    packet2.release();
                }
            } catch (RuntimeException e3) {
                if (this.asyncErrorListener == null) {
                    throw e3;
                }
                this.asyncErrorListener.onError(e3);
                if (0 != 0) {
                    packet.release();
                }
            }
        } catch (Throwable th) {
            if (0 != 0) {
                packet.release();
            }
            throw th;
        }
    }

    public void waitUntilIdle() {
        try {
            this.mediapipeGraph.waitUntilGraphIdle();
        } catch (MediaPipeException e2) {
            if (this.asyncErrorListener != null) {
                this.asyncErrorListener.onError(e2);
            } else {
                Log.e("FrameProcessor", "Mediapipe error: ", e2);
            }
        }
    }

    private void startGraph() {
        this.mediapipeGraph.startRunningGraph();
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[]}, finally: {[INVOKE, IF] complete} */
    /* JADX DEBUG: Don't trust debug lines info. Repeating lines: [579=4, 582=4] */
    @Override // com.google.mediapipe.components.AudioDataConsumer
    public void onNewAudioData(ByteBuffer audioData, long timestampMicros, AudioFormat audioFormat) {
        Packet audioPacket = null;
        try {
            try {
                if (!this.started.getAndSet(true)) {
                    startGraph();
                }
                if (audioFormat.getChannelCount() != this.numAudioChannels || audioFormat.getSampleRate() != this.audioSampleRate || audioFormat.getEncoding() != 2) {
                    Log.e("FrameProcessor", "Producer's AudioFormat doesn't match FrameProcessor's AudioFormat");
                    if (0 != 0) {
                        audioPacket.release();
                        return;
                    }
                    return;
                }
                Preconditions.checkNotNull(this.audioInputStream);
                int numSamples = (audioData.limit() / 2) / this.numAudioChannels;
                Packet audioPacket2 = this.packetCreator.createAudioPacket(audioData, this.numAudioChannels, numSamples);
                try {
                    this.mediapipeGraph.addConsumablePacketToInputStream(this.audioInputStream, audioPacket2, timestampMicros);
                    audioPacket2 = null;
                } catch (MediaPipeException e2) {
                    if (this.asyncErrorListener != null) {
                        throw e2;
                    }
                    Log.e("FrameProcessor", "Mediapipe error: ", e2);
                }
                if (audioPacket2 != null) {
                    audioPacket2.release();
                }
            } catch (RuntimeException e3) {
                if (this.asyncErrorListener == null) {
                    throw e3;
                }
                this.asyncErrorListener.onError(e3);
                if (0 != 0) {
                    audioPacket.release();
                }
            }
        } catch (Throwable th) {
            if (0 != 0) {
                audioPacket.release();
            }
            throw th;
        }
    }

    public void addAudioConsumer(AudioDataConsumer consumer) {
        synchronized (this) {
            List<AudioDataConsumer> newConsumers = new ArrayList<>(this.audioConsumers);
            newConsumers.add(consumer);
            this.audioConsumers = newConsumers;
        }
    }

    public boolean removeAudioConsumer(AudioDataConsumer consumer) {
        boolean existed;
        synchronized (this) {
            List<AudioDataConsumer> newConsumers = new ArrayList<>(this.audioConsumers);
            existed = newConsumers.remove(consumer);
            this.audioConsumers = newConsumers;
        }
        return existed;
    }
}