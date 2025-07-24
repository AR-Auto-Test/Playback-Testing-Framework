package com.google.mediapipe.framework;

import android.graphics.Bitmap;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AndroidPacketCreator.class */
public class AndroidPacketCreator extends PacketCreator {
    private native long nativeCreateRgbImageFrame(long context, Bitmap bitmap);

    private native long nativeCreateRgbaImageFrame(long context, Bitmap bitmap);

    public AndroidPacketCreator(Graph context) {
        super(context);
    }

    public Packet createRgbImageFrame(Bitmap bitmap) {
        if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            throw new RuntimeException("bitmap must use ARGB_8888 config.");
        }
        return Packet.create(nativeCreateRgbImageFrame(this.mediapipeGraph.getNativeHandle(), bitmap));
    }

    public Packet createRgbaImageFrame(Bitmap bitmap) {
        if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            throw new RuntimeException("bitmap must use ARGB_8888 config.");
        }
        return Packet.create(nativeCreateRgbaImageFrame(this.mediapipeGraph.getNativeHandle(), bitmap));
    }
}