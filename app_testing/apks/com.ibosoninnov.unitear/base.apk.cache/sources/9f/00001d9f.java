package com.google.ar.schemas.lull;

import com.google.common.primitives.UnsignedInts;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class ModelIndexRange extends Struct {
    public static int createModelIndexRange(FlatBufferBuilder flatBufferBuilder, long j, long j2) {
        flatBufferBuilder.prep(4, 8);
        flatBufferBuilder.putInt((int) j2);
        flatBufferBuilder.putInt((int) j);
        return flatBufferBuilder.offset();
    }

    public ModelIndexRange __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public long end() {
        return this.bb.getInt(this.bb_pos + 4) & UnsignedInts.INT_MASK;
    }

    public long start() {
        return this.bb.getInt(this.bb_pos + 0) & UnsignedInts.INT_MASK;
    }
}