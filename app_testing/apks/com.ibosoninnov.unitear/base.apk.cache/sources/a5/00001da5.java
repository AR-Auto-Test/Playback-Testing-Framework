package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelPipelineRenderableDef extends Table {
    public static void addAttributes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addMaterials(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addSource(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createAttributesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addInt(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createMaterialsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createModelPipelineRenderableDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3) {
        flatBufferBuilder.startObject(3);
        addAttributes(flatBufferBuilder, i3);
        addMaterials(flatBufferBuilder, i2);
        addSource(flatBufferBuilder, i);
        return endModelPipelineRenderableDef(flatBufferBuilder);
    }

    public static int endModelPipelineRenderableDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelPipelineRenderableDef getRootAsModelPipelineRenderableDef(ByteBuffer byteBuffer) {
        return getRootAsModelPipelineRenderableDef(byteBuffer, new ModelPipelineRenderableDef());
    }

    public static void startAttributesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startMaterialsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startModelPipelineRenderableDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(3);
    }

    public ModelPipelineRenderableDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int attributes(int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return this.bb.getInt((i * 4) + __vector(__offset));
        }
        return 0;
    }

    public ByteBuffer attributesAsByteBuffer() {
        return __vector_as_bytebuffer(8, 4);
    }

    public ByteBuffer attributesInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 8, 4);
    }

    public int attributesLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public ModelPipelineMaterialDef materials(int i) {
        return materials(new ModelPipelineMaterialDef(), i);
    }

    public int materialsLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public String source() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer sourceAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer sourceInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public static ModelPipelineRenderableDef getRootAsModelPipelineRenderableDef(ByteBuffer byteBuffer, ModelPipelineRenderableDef modelPipelineRenderableDef) {
        return modelPipelineRenderableDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public ModelPipelineMaterialDef materials(ModelPipelineMaterialDef modelPipelineMaterialDef, int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return modelPipelineMaterialDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}