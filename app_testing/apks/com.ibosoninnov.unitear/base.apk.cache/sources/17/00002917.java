package com.google.mediapipe.framework;

import com.google.mediapipe.framework.ProtoUtil;
import com.google.protobuf.MessageLite;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/PacketCreator.class */
public class PacketCreator {
    protected Graph mediapipeGraph;

    private native long nativeCreateReferencePacket(long context, long packet);

    private native long nativeCreateRgbImage(long context, ByteBuffer buffer, int width, int height);

    private native long nativeCreateAudioPacket(long context, byte[] data, int offset, int numChannels, int numSamples);

    private native long nativeCreateAudioPacketDirect(long context, ByteBuffer data, int numChannels, int numSamples);

    private native long nativeCreateRgbImageFromRgba(long context, ByteBuffer buffer, int width, int height);

    private native long nativeCreateGrayscaleImage(long context, ByteBuffer buffer, int width, int height);

    private native long nativeCreateRgbaImageFrame(long context, ByteBuffer buffer, int width, int height);

    private native long nativeCreateFloatImageFrame(long context, FloatBuffer buffer, int width, int height);

    private native long nativeCreateInt16(long context, short value);

    private native long nativeCreateInt32(long context, int value);

    private native long nativeCreateInt64(long context, long value);

    private native long nativeCreateFloat32(long context, float value);

    private native long nativeCreateFloat64(long context, double value);

    private native long nativeCreateBool(long context, boolean value);

    private native long nativeCreateString(long context, String value);

    private native long nativeCreateVideoHeader(long context, int width, int height);

    private native long nativeCreateTimeSeriesHeader(long context, int numChannels, double sampleRate);

    private native long nativeCreateMatrix(long context, int rows, int cols, float[] data);

    private native long nativeCreateGpuBuffer(long context, int name, int width, int height, TextureReleaseCallback releaseCallback);

    private native long nativeCreateInt32Array(long context, int[] data);

    private native long nativeCreateFloat32Array(long context, float[] data);

    private native long nativeCreateFloat32Vector(long context, float[] data);

    private native long nativeCreateStringFromByteArray(long context, byte[] data);

    private native long nativeCreateProto(long context, ProtoUtil.SerializedMessage data);

    private native long nativeCreateCalculatorOptions(long context, byte[] data);

    private native long nativeCreateCameraIntrinsics(long context, float fx, float fy, float cx, float cy, float width, float height);

    public PacketCreator(Graph context) {
        this.mediapipeGraph = context;
    }

    public Packet createReferencePacket(Packet packet) {
        return Packet.create(nativeCreateReferencePacket(this.mediapipeGraph.getNativeHandle(), packet.getNativeHandle()));
    }

    public Packet createRgbImage(ByteBuffer buffer, int width, int height) {
        int widthStep = (((width * 3) + 3) / 4) * 4;
        if (widthStep * height != buffer.capacity()) {
            throw new RuntimeException("The size of the buffer should be: " + (widthStep * height));
        }
        return Packet.create(nativeCreateRgbImage(this.mediapipeGraph.getNativeHandle(), buffer, width, height));
    }

    public Packet createAudioPacket(byte[] data, int numChannels, int numSamples) {
        checkAudioDataSize(data.length, numChannels, numSamples);
        return Packet.create(nativeCreateAudioPacket(this.mediapipeGraph.getNativeHandle(), data, 0, numChannels, numSamples));
    }

    public Packet createAudioPacket(ByteBuffer data, int numChannels, int numSamples) {
        checkAudioDataSize(data.remaining(), numChannels, numSamples);
        if (data.isDirect()) {
            return Packet.create(nativeCreateAudioPacketDirect(this.mediapipeGraph.getNativeHandle(), data.slice(), numChannels, numSamples));
        }
        if (data.hasArray()) {
            return Packet.create(nativeCreateAudioPacket(this.mediapipeGraph.getNativeHandle(), data.array(), data.arrayOffset() + data.position(), numChannels, numSamples));
        }
        throw new IllegalArgumentException("Data must be either a direct byte buffer or be backed by a byte array.");
    }

    private static void checkAudioDataSize(int length, int numChannels, int numSamples) {
        int expectedLength = numChannels * numSamples * 2;
        if (expectedLength != length) {
            throw new IllegalArgumentException("Please check the audio data size, has to be num_channels * num_samples * 2 = " + expectedLength + " but was " + length);
        }
    }

    public Packet createRgbImageFromRgba(ByteBuffer buffer, int width, int height) {
        if (width * height * 4 != buffer.capacity()) {
            throw new RuntimeException("The size of the buffer should be: " + (width * height * 4));
        }
        return Packet.create(nativeCreateRgbImageFromRgba(this.mediapipeGraph.getNativeHandle(), buffer, width, height));
    }

    public Packet createGrayscaleImage(ByteBuffer buffer, int width, int height) {
        if (width * height != buffer.capacity()) {
            throw new RuntimeException("The size of the buffer should be: " + (width * height) + " but is " + buffer.capacity());
        }
        return Packet.create(nativeCreateGrayscaleImage(this.mediapipeGraph.getNativeHandle(), buffer, width, height));
    }

    public Packet createRgbaImageFrame(ByteBuffer buffer, int width, int height) {
        if (buffer.capacity() != width * height * 4) {
            throw new RuntimeException("buffer doesn't have the correct size.");
        }
        return Packet.create(nativeCreateRgbaImageFrame(this.mediapipeGraph.getNativeHandle(), buffer, width, height));
    }

    public Packet createFloatImageFrame(FloatBuffer buffer, int width, int height) {
        if (buffer.capacity() != width * height * 4) {
            throw new RuntimeException("buffer doesn't have the correct size.");
        }
        return Packet.create(nativeCreateFloatImageFrame(this.mediapipeGraph.getNativeHandle(), buffer, width, height));
    }

    public Packet createInt16(short value) {
        return Packet.create(nativeCreateInt16(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createInt32(int value) {
        return Packet.create(nativeCreateInt32(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createInt64(long value) {
        return Packet.create(nativeCreateInt64(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createFloat32(float value) {
        return Packet.create(nativeCreateFloat32(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createFloat64(double value) {
        return Packet.create(nativeCreateFloat64(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createBool(boolean value) {
        return Packet.create(nativeCreateBool(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createString(String value) {
        return Packet.create(nativeCreateString(this.mediapipeGraph.getNativeHandle(), value));
    }

    public Packet createInt16Vector(short[] data) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public Packet createInt32Vector(int[] data) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public Packet createInt64Vector(long[] data) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public Packet createFloat32Vector(float[] data) {
        return Packet.create(nativeCreateFloat32Vector(this.mediapipeGraph.getNativeHandle(), data));
    }

    public Packet createFloat64Vector(double[] data) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public Packet createInt32Array(int[] data) {
        return Packet.create(nativeCreateInt32Array(this.mediapipeGraph.getNativeHandle(), data));
    }

    public Packet createFloat32Array(float[] data) {
        return Packet.create(nativeCreateFloat32Array(this.mediapipeGraph.getNativeHandle(), data));
    }

    public Packet createByteArray(byte[] data) {
        return Packet.create(nativeCreateStringFromByteArray(this.mediapipeGraph.getNativeHandle(), data));
    }

    public Packet createVideoHeader(int width, int height) {
        return Packet.create(nativeCreateVideoHeader(this.mediapipeGraph.getNativeHandle(), width, height));
    }

    public Packet createTimeSeriesHeader(int numChannels, double sampleRate) {
        return Packet.create(nativeCreateTimeSeriesHeader(this.mediapipeGraph.getNativeHandle(), numChannels, sampleRate));
    }

    public Packet createMatrix(int rows, int cols, float[] data) {
        return Packet.create(nativeCreateMatrix(this.mediapipeGraph.getNativeHandle(), rows, cols, data));
    }

    public Packet createSerializedProto(MessageLite message) {
        return Packet.create(nativeCreateStringFromByteArray(this.mediapipeGraph.getNativeHandle(), message.toByteArray()));
    }

    public Packet createCalculatorOptions(MessageLite message) {
        return Packet.create(nativeCreateCalculatorOptions(this.mediapipeGraph.getNativeHandle(), message.toByteArray()));
    }

    public Packet createProto(MessageLite message) {
        ProtoUtil.SerializedMessage serialized = ProtoUtil.pack(message);
        return Packet.create(nativeCreateProto(this.mediapipeGraph.getNativeHandle(), serialized));
    }

    public Packet createCameraIntrinsics(float fx, float fy, float cx, float cy, float width, float height) {
        return Packet.create(nativeCreateCameraIntrinsics(this.mediapipeGraph.getNativeHandle(), fx, fy, cx, cy, width, height));
    }

    public Packet createGpuBuffer(int name, int width, int height, TextureReleaseCallback releaseCallback) {
        return Packet.create(nativeCreateGpuBuffer(this.mediapipeGraph.getNativeHandle(), name, width, height, releaseCallback));
    }

    @Deprecated
    public Packet createGpuBuffer(int name, int width, int height) {
        return Packet.create(nativeCreateGpuBuffer(this.mediapipeGraph.getNativeHandle(), name, width, height, null));
    }

    public Packet createGpuBuffer(TextureFrame frame) {
        return Packet.create(nativeCreateGpuBuffer(this.mediapipeGraph.getNativeHandle(), frame.getTextureName(), frame.getWidth(), frame.getHeight(), frame));
    }

    private void releaseWithSyncToken(long nativeSyncToken, TextureReleaseCallback releaseCallback) {
        releaseCallback.release(new GraphGlSyncToken(nativeSyncToken));
    }
}