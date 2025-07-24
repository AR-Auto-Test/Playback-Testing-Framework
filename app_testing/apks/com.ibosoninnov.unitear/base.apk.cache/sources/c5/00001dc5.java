package com.google.ar.schemas.motive;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class CompactSplineFloatNodeFb extends Struct {
    public static int createCompactSplineFloatNodeFb(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4) {
        flatBufferBuilder.prep(4, 12);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public CompactSplineFloatNodeFb __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public float derivative() {
        return this.bb.getFloat(this.bb_pos + 8);
    }

    public float time() {
        return this.bb.getFloat(this.bb_pos + 4);
    }

    public float value() {
        return this.bb.getFloat(this.bb_pos + 0);
    }
}