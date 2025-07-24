package com.google.mediapipe.framework;

import com.google.common.base.Preconditions;
import com.google.common.flogger.FluentLogger;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLite;
import com.google.protobuf.Parser;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/PacketGetter.class */
public final class PacketGetter {
    private static final FluentLogger logger = FluentLogger.forEnclosingClass();

    private static native long nativeGetPacketFromReference(long nativePacketHandle);

    private static native long[] nativeGetPairPackets(long nativePacketHandle);

    private static native long[] nativeGetVectorPackets(long nativePacketHandle);

    private static native short nativeGetInt16(long nativePacketHandle);

    private static native int nativeGetInt32(long nativePacketHandle);

    private static native long nativeGetInt64(long nativePacketHandle);

    private static native float nativeGetFloat32(long nativePacketHandle);

    private static native double nativeGetFloat64(long nativePacketHandle);

    private static native boolean nativeGetBool(long nativePacketHandle);

    private static native String nativeGetString(long nativePacketHandle);

    private static native byte[] nativeGetBytes(long nativePacketHandle);

    private static native byte[] nativeGetProtoBytes(long nativePacketHandle);

    private static native void nativeGetProto(long nativePacketHandle, ProtoUtil.SerializedMessage result);

    private static native short[] nativeGetInt16Vector(long nativePacketHandle);

    private static native int[] nativeGetInt32Vector(long nativePacketHandle);

    private static native long[] nativeGetInt64Vector(long nativePacketHandle);

    private static native float[] nativeGetFloat32Vector(long nativePacketHandle);

    private static native double[] nativeGetFloat64Vector(long nativePacketHandle);

    private static native byte[][] nativeGetProtoVector(long nativePacketHandle);

    private static native int nativeGetImageWidth(long nativePacketHandle);

    private static native int nativeGetImageHeight(long nativePacketHandle);

    private static native boolean nativeGetImageData(long nativePacketHandle, ByteBuffer buffer);

    private static native boolean nativeGetRgbaFromRgb(long nativePacketHandle, ByteBuffer buffer);

    private static native int nativeGetVideoHeaderWidth(long nativepackethandle);

    private static native int nativeGetVideoHeaderHeight(long nativepackethandle);

    private static native int nativeGetTimeSeriesHeaderNumChannels(long nativepackethandle);

    private static native double nativeGetTimeSeriesHeaderSampleRate(long nativepackethandle);

    private static native byte[] nativeGetAudioData(long nativePacketHandle);

    private static native float[] nativeGetMatrixData(long nativePacketHandle);

    private static native int nativeGetMatrixRows(long nativePacketHandle);

    private static native int nativeGetMatrixCols(long nativePacketHandle);

    private static native int nativeGetGpuBufferName(long nativePacketHandle);

    private static native long nativeGetGpuBuffer(long nativePacketHandle);

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/PacketGetter$PacketPair.class */
    public static class PacketPair {
        final Packet first;
        final Packet second;

        public PacketPair(Packet first, Packet second) {
            this.first = first;
            this.second = second;
        }
    }

    public static Packet getPacketFromReference(final Packet referencePacket) {
        return Packet.create(nativeGetPacketFromReference(referencePacket.getNativeHandle()));
    }

    public static PacketPair getPairOfPackets(final Packet packet) {
        long[] handles = nativeGetPairPackets(packet.getNativeHandle());
        return new PacketPair(Packet.create(handles[0]), Packet.create(handles[1]));
    }

    public static List<Packet> getVectorOfPackets(final Packet packet) {
        long[] handles = nativeGetVectorPackets(packet.getNativeHandle());
        List<Packet> packets = new ArrayList<>(handles.length);
        for (long handle : handles) {
            packets.add(Packet.create(handle));
        }
        return packets;
    }

    public static short getInt16(final Packet packet) {
        return nativeGetInt16(packet.getNativeHandle());
    }

    public static int getInt32(final Packet packet) {
        return nativeGetInt32(packet.getNativeHandle());
    }

    public static long getInt64(final Packet packet) {
        return nativeGetInt64(packet.getNativeHandle());
    }

    public static float getFloat32(final Packet packet) {
        return nativeGetFloat32(packet.getNativeHandle());
    }

    public static double getFloat64(final Packet packet) {
        return nativeGetFloat64(packet.getNativeHandle());
    }

    public static boolean getBool(final Packet packet) {
        return nativeGetBool(packet.getNativeHandle());
    }

    public static String getString(final Packet packet) {
        return nativeGetString(packet.getNativeHandle());
    }

    public static byte[] getBytes(final Packet packet) {
        return nativeGetBytes(packet.getNativeHandle());
    }

    public static byte[] getProtoBytes(final Packet packet) {
        return nativeGetProtoBytes(packet.getNativeHandle());
    }

    public static <T extends MessageLite> T getProto(final Packet packet, Class<T> clazz) throws InvalidProtocolBufferException {
        ProtoUtil.SerializedMessage result = new ProtoUtil.SerializedMessage();
        nativeGetProto(packet.getNativeHandle(), result);
        return (T) ProtoUtil.unpack(result, clazz);
    }

    public static short[] getInt16Vector(final Packet packet) {
        return nativeGetInt16Vector(packet.getNativeHandle());
    }

    public static int[] getInt32Vector(final Packet packet) {
        return nativeGetInt32Vector(packet.getNativeHandle());
    }

    public static long[] getInt64Vector(final Packet packet) {
        return nativeGetInt64Vector(packet.getNativeHandle());
    }

    public static float[] getFloat32Vector(final Packet packet) {
        return nativeGetFloat32Vector(packet.getNativeHandle());
    }

    public static double[] getFloat64Vector(final Packet packet) {
        return nativeGetFloat64Vector(packet.getNativeHandle());
    }

    public static <T> List<T> getProtoVector(final Packet packet, Parser<T> messageParser) {
        byte[][] protoVector = nativeGetProtoVector(packet.getNativeHandle());
        Preconditions.checkNotNull(protoVector, "Vector of protocol buffer objects should not be null!");
        try {
            List<T> parsedMessageList = new ArrayList<>();
            for (byte[] message : protoVector) {
                T parsedMessage = messageParser.parseFrom(message);
                parsedMessageList.add(parsedMessage);
            }
            return parsedMessageList;
        } catch (InvalidProtocolBufferException e2) {
            throw new IllegalArgumentException(e2);
        }
    }

    public static int getImageWidth(final Packet packet) {
        return nativeGetImageWidth(packet.getNativeHandle());
    }

    public static int getImageHeight(final Packet packet) {
        return nativeGetImageHeight(packet.getNativeHandle());
    }

    public static boolean getImageData(final Packet packet, ByteBuffer buffer) {
        return nativeGetImageData(packet.getNativeHandle(), buffer);
    }

    public static boolean getRgbaFromRgb(final Packet packet, ByteBuffer buffer) {
        return nativeGetRgbaFromRgb(packet.getNativeHandle(), buffer);
    }

    public static byte[] getAudioByteData(final Packet packet) {
        return nativeGetAudioData(packet.getNativeHandle());
    }

    public static int getAudioDataNumChannels(final Packet packet) {
        return nativeGetMatrixRows(packet.getNativeHandle());
    }

    public static int getAudioDataNumSamples(final Packet packet) {
        return nativeGetMatrixCols(packet.getNativeHandle());
    }

    public static int getTimeSeriesHeaderNumChannels(final Packet packet) {
        return nativeGetTimeSeriesHeaderNumChannels(packet.getNativeHandle());
    }

    public static double getTimeSeriesHeaderSampleRate(final Packet packet) {
        return nativeGetTimeSeriesHeaderSampleRate(packet.getNativeHandle());
    }

    public static int getVideoHeaderWidth(final Packet packet) {
        return nativeGetVideoHeaderWidth(packet.getNativeHandle());
    }

    public static int getVideoHeaderHeight(final Packet packet) {
        return nativeGetVideoHeaderHeight(packet.getNativeHandle());
    }

    public static float[] getMatrixData(final Packet packet) {
        return nativeGetMatrixData(packet.getNativeHandle());
    }

    public static int getMatrixRows(final Packet packet) {
        return nativeGetMatrixRows(packet.getNativeHandle());
    }

    public static int getMatrixCols(final Packet packet) {
        return nativeGetMatrixCols(packet.getNativeHandle());
    }

    @Deprecated
    public static int getGpuBufferName(final Packet packet) {
        return nativeGetGpuBufferName(packet.getNativeHandle());
    }

    public static GraphTextureFrame getTextureFrame(final Packet packet) {
        return new GraphTextureFrame(nativeGetGpuBuffer(packet.getNativeHandle()), packet.getTimestamp());
    }

    private PacketGetter() {
    }
}