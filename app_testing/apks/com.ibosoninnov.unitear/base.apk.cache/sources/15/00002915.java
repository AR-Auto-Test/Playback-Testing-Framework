package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/Packet.class */
public class Packet {
    private long nativePacketHandle;

    private native void nativeReleasePacket(long packetHandle);

    private native long nativeCopyPacket(long packetHandle);

    private native long nativeGetTimestamp(long packetHandle);

    public static Packet create(long nativeHandle) {
        return new Packet(nativeHandle);
    }

    public long getNativeHandle() {
        return this.nativePacketHandle;
    }

    public long getTimestamp() {
        return nativeGetTimestamp(this.nativePacketHandle);
    }

    public Packet copy() {
        return new Packet(nativeCopyPacket(this.nativePacketHandle));
    }

    public void release() {
        if (this.nativePacketHandle != 0) {
            nativeReleasePacket(this.nativePacketHandle);
            this.nativePacketHandle = 0L;
        }
    }

    private Packet(long handle) {
        this.nativePacketHandle = handle;
    }
}