package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.ar.schemas.lull.Vec3;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class TransformDef extends Table {
    public static void addOffset(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(1, i, 0);
    }

    public static void addScale(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int endTransformDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static TransformDef getRootAsTransformDef(ByteBuffer byteBuffer) {
        return getRootAsTransformDef(byteBuffer, new TransformDef());
    }

    public static void startTransformDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public TransformDef __assign(int i, ByteBuffer byteBuffer) {
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

    public Vec3 offset() {
        return offset(new Vec3());
    }

    public float scale() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static TransformDef getRootAsTransformDef(ByteBuffer byteBuffer, TransformDef transformDef) {
        return transformDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Vec3 offset(Vec3 vec3) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}