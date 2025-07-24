package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelPipelineMaterialDef extends Table {
    public static void addMaterial(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addNameOverride(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static int createModelPipelineMaterialDef(FlatBufferBuilder flatBufferBuilder, int i, int i2) {
        flatBufferBuilder.startObject(2);
        addNameOverride(flatBufferBuilder, i2);
        addMaterial(flatBufferBuilder, i);
        return endModelPipelineMaterialDef(flatBufferBuilder);
    }

    public static int endModelPipelineMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelPipelineMaterialDef getRootAsModelPipelineMaterialDef(ByteBuffer byteBuffer) {
        return getRootAsModelPipelineMaterialDef(byteBuffer, new ModelPipelineMaterialDef());
    }

    public static void startModelPipelineMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public ModelPipelineMaterialDef __assign(int i, ByteBuffer byteBuffer) {
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

    public MaterialDef material() {
        return material(new MaterialDef());
    }

    public String nameOverride() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer nameOverrideAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer nameOverrideInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 1);
    }

    public static ModelPipelineMaterialDef getRootAsModelPipelineMaterialDef(ByteBuffer byteBuffer, ModelPipelineMaterialDef modelPipelineMaterialDef) {
        return modelPipelineMaterialDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public MaterialDef material(MaterialDef materialDef) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return materialDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}