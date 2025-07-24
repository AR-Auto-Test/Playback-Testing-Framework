package com.google.ar.schemas.lull;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class AabbDef extends Struct {
    public static int createAabbDef(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4, float f5, float f6, float f7) {
        flatBufferBuilder.prep(4, 24);
        flatBufferBuilder.prep(4, 12);
        flatBufferBuilder.putFloat(f7);
        flatBufferBuilder.putFloat(f6);
        flatBufferBuilder.putFloat(f5);
        flatBufferBuilder.prep(4, 12);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public AabbDef __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public Vec3 max() {
        return max(new Vec3());
    }

    public Vec3 min() {
        return min(new Vec3());
    }

    public Vec3 max(Vec3 vec3) {
        return vec3.__assign(this.bb_pos + 12, this.bb);
    }

    public Vec3 min(Vec3 vec3) {
        return vec3.__assign(this.bb_pos + 0, this.bb);
    }
}