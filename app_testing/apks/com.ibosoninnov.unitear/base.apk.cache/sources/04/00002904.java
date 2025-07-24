package com.google.mediapipe.framework;

import android.graphics.Bitmap;
import com.google.common.base.Preconditions;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AndroidPacketGetter.class */
public final class AndroidPacketGetter {
    public static Bitmap getBitmapFromRgb(Packet packet) {
        int width = PacketGetter.getImageWidth(packet);
        int height = PacketGetter.getImageHeight(packet);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        copyRgbToBitmap(packet, bitmap, width, height);
        return bitmap;
    }

    public static void copyRgbToBitmap(Packet packet, Bitmap inBitmap) {
        Preconditions.checkArgument(inBitmap.isMutable(), "Input bitmap should be mutable.");
        Preconditions.checkArgument(inBitmap.getConfig() == Bitmap.Config.ARGB_8888, "Input bitmap should be of type ARGB_8888.");
        int width = PacketGetter.getImageWidth(packet);
        int height = PacketGetter.getImageHeight(packet);
        Preconditions.checkArgument(inBitmap.getByteCount() == (width * height) * 4, "Input bitmap size mismatch.");
        copyRgbToBitmap(packet, inBitmap, width, height);
    }

    private static void copyRgbToBitmap(Packet packet, Bitmap mutableBitmap, int width, int height) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
        PacketGetter.getRgbaFromRgb(packet, buffer);
        mutableBitmap.copyPixelsFromBuffer(buffer);
    }

    public static Bitmap getBitmapFromRgba(Packet packet) {
        int width = PacketGetter.getImageWidth(packet);
        int height = PacketGetter.getImageHeight(packet);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        copyRgbaToBitmap(packet, bitmap, width, height);
        return bitmap;
    }

    public static void copyRgbaToBitmap(Packet packet, Bitmap inBitmap) {
        Preconditions.checkArgument(inBitmap.isMutable(), "Input bitmap should be mutable.");
        Preconditions.checkArgument(inBitmap.getConfig() == Bitmap.Config.ARGB_8888, "Input bitmap should be of type ARGB_8888.");
        int width = PacketGetter.getImageWidth(packet);
        int height = PacketGetter.getImageHeight(packet);
        Preconditions.checkArgument(inBitmap.getByteCount() == (width * height) * 4, "Input bitmap size mismatch.");
        copyRgbaToBitmap(packet, inBitmap, width, height);
    }

    private static void copyRgbaToBitmap(Packet packet, Bitmap mutableBitmap, int width, int height) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
        buffer.order(ByteOrder.nativeOrder());
        boolean status = PacketGetter.getImageData(packet, buffer);
        Preconditions.checkState(status, String.format("Got error from getImageData, returning null Bitmap. Image width %d, height %d", Integer.valueOf(width), Integer.valueOf(height)));
        mutableBitmap.copyPixelsFromBuffer(buffer);
    }

    private AndroidPacketGetter() {
    }
}