package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class CompactSplineAnimFloatFb extends Table {
    public static boolean CompactSplineAnimFloatFbBufferHasIdentifier(ByteBuffer byteBuffer) {
        return Table.__has_identifier(byteBuffer, "SPLN");
    }

    public static void addSplines(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createCompactSplineAnimFloatFb(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addSplines(flatBufferBuilder, i);
        return endCompactSplineAnimFloatFb(flatBufferBuilder);
    }

    public static int createSplinesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endCompactSplineAnimFloatFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishCompactSplineAnimFloatFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i, "SPLN");
    }

    public static void finishSizePrefixedCompactSplineAnimFloatFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i, "SPLN");
    }

    public static CompactSplineAnimFloatFb getRootAsCompactSplineAnimFloatFb(ByteBuffer byteBuffer) {
        return getRootAsCompactSplineAnimFloatFb(byteBuffer, new CompactSplineAnimFloatFb());
    }

    public static void startCompactSplineAnimFloatFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public static void startSplinesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public CompactSplineAnimFloatFb __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
        int i2 = i - byteBuffer.getInt(i);
        this.vtable_start = i2;
        this.vtable_size = this.bb.getShort(i2);
    }

    public CompactSplineFloatFb splines(int i) {
        return splines(new CompactSplineFloatFb(), i);
    }

    public int splinesLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static CompactSplineAnimFloatFb getRootAsCompactSplineAnimFloatFb(ByteBuffer byteBuffer, CompactSplineAnimFloatFb compactSplineAnimFloatFb) {
        return compactSplineAnimFloatFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public CompactSplineFloatFb splines(CompactSplineFloatFb compactSplineFloatFb, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return compactSplineFloatFb.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}