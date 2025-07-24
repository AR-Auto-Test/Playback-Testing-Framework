package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class CompactSplineFloatFb extends Table {
    public static void addMaxValue(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addMinValue(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addNodes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static int createCompactSplineFloatFb(FlatBufferBuilder flatBufferBuilder, float f2, float f3, int i) {
        flatBufferBuilder.startObject(3);
        addNodes(flatBufferBuilder, i);
        addMaxValue(flatBufferBuilder, f3);
        addMinValue(flatBufferBuilder, f2);
        return endCompactSplineFloatFb(flatBufferBuilder);
    }

    public static int endCompactSplineFloatFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static CompactSplineFloatFb getRootAsCompactSplineFloatFb(ByteBuffer byteBuffer) {
        return getRootAsCompactSplineFloatFb(byteBuffer, new CompactSplineFloatFb());
    }

    public static void startCompactSplineFloatFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(3);
    }

    public static void startNodesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(12, i, 4);
    }

    public CompactSplineFloatFb __assign(int i, ByteBuffer byteBuffer) {
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

    public float maxValue() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float minValue() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public CompactSplineFloatNodeFb nodes(int i) {
        return nodes(new CompactSplineFloatNodeFb(), i);
    }

    public int nodesLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static CompactSplineFloatFb getRootAsCompactSplineFloatFb(ByteBuffer byteBuffer, CompactSplineFloatFb compactSplineFloatFb) {
        return compactSplineFloatFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public CompactSplineFloatNodeFb nodes(CompactSplineFloatNodeFb compactSplineFloatNodeFb, int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return compactSplineFloatNodeFb.__assign((i * 12) + __vector(__offset), this.bb);
        }
        return null;
    }
}