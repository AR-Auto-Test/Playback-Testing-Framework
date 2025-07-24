package com.google.ar.schemas.motive;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class CompactSplineNodeFb extends Struct {
    public static int createCompactSplineNodeFb(FlatBufferBuilder flatBufferBuilder, int i, int i2, short s) {
        flatBufferBuilder.prep(2, 6);
        flatBufferBuilder.putShort(s);
        flatBufferBuilder.putShort((short) i2);
        flatBufferBuilder.putShort((short) i);
        return flatBufferBuilder.offset();
    }

    public CompactSplineNodeFb __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public short angle() {
        return this.bb.getShort(this.bb_pos + 4);
    }

    public int x() {
        return this.bb.getShort(this.bb_pos + 0) & 65535;
    }

    public int y() {
        return this.bb.getShort(this.bb_pos + 2) & 65535;
    }
}