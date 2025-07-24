package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class CompactSplineFb extends Table {
    public static void addNodes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addXGranularity(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(2, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addYRangeEnd(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addYRangeStart(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int createCompactSplineFb(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4, int i) {
        flatBufferBuilder.startObject(4);
        addNodes(flatBufferBuilder, i);
        addXGranularity(flatBufferBuilder, f4);
        addYRangeEnd(flatBufferBuilder, f3);
        addYRangeStart(flatBufferBuilder, f2);
        return endCompactSplineFb(flatBufferBuilder);
    }

    public static int endCompactSplineFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static CompactSplineFb getRootAsCompactSplineFb(ByteBuffer byteBuffer) {
        return getRootAsCompactSplineFb(byteBuffer, new CompactSplineFb());
    }

    public static void startCompactSplineFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public static void startNodesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(6, i, 2);
    }

    public CompactSplineFb __assign(int i, ByteBuffer byteBuffer) {
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

    public CompactSplineNodeFb nodes(int i) {
        return nodes(new CompactSplineNodeFb(), i);
    }

    public int nodesLength() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public float xGranularity() {
        int __offset = __offset(8);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float yRangeEnd() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float yRangeStart() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static CompactSplineFb getRootAsCompactSplineFb(ByteBuffer byteBuffer, CompactSplineFb compactSplineFb) {
        return compactSplineFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public CompactSplineNodeFb nodes(CompactSplineNodeFb compactSplineNodeFb, int i) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return compactSplineNodeFb.__assign((i * 6) + __vector(__offset), this.bb);
        }
        return null;
    }
}