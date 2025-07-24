package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.common.primitives.UnsignedInts;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelInstanceDef extends Table {
    public static void addAabbs(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(11, i, 0);
    }

    public static void addBlendAttributes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(10, i, 0);
    }

    public static void addBlendShapes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(9, i, 0);
    }

    public static void addIndices16(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addIndices32(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addInterleaved(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(7, z, true);
    }

    public static void addMaterials(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static void addNumVertices(FlatBufferBuilder flatBufferBuilder, long j) {
        flatBufferBuilder.addInt(6, (int) j, 0);
    }

    public static void addRanges(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addShaderToMeshBones(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(8, i, 0);
    }

    public static void addVertexAttributes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(5, i, 0);
    }

    public static void addVertexData(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createAabbsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createBlendShapesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createIndices16Vector(FlatBufferBuilder flatBufferBuilder, short[] sArr) {
        flatBufferBuilder.startVector(2, sArr.length, 2);
        for (int length = sArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addShort(sArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createIndices32Vector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
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

    public static int createModelInstanceDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4, int i5, int i6, long j, boolean z, int i7, int i8, int i9, int i10) {
        flatBufferBuilder.startObject(12);
        addAabbs(flatBufferBuilder, i10);
        addBlendAttributes(flatBufferBuilder, i9);
        addBlendShapes(flatBufferBuilder, i8);
        addShaderToMeshBones(flatBufferBuilder, i7);
        addNumVertices(flatBufferBuilder, j);
        addVertexAttributes(flatBufferBuilder, i6);
        addMaterials(flatBufferBuilder, i5);
        addRanges(flatBufferBuilder, i4);
        addIndices32(flatBufferBuilder, i3);
        addIndices16(flatBufferBuilder, i2);
        addVertexData(flatBufferBuilder, i);
        addInterleaved(flatBufferBuilder, z);
        return endModelInstanceDef(flatBufferBuilder);
    }

    public static int createShaderToMeshBonesVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int createVertexDataVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int endModelInstanceDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelInstanceDef getRootAsModelInstanceDef(ByteBuffer byteBuffer) {
        return getRootAsModelInstanceDef(byteBuffer, new ModelInstanceDef());
    }

    public static void startAabbsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startBlendAttributesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(8, i, 4);
    }

    public static void startBlendShapesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startIndices16Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(2, i, 2);
    }

    public static void startIndices32Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startMaterialsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startModelInstanceDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(12);
    }

    public static void startRangesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(8, i, 4);
    }

    public static void startShaderToMeshBonesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startVertexAttributesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(8, i, 4);
    }

    public static void startVertexDataVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public ModelInstanceDef __assign(int i, ByteBuffer byteBuffer) {
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

    public SubmeshAabb aabbs(int i) {
        return aabbs(new SubmeshAabb(), i);
    }

    public int aabbsLength() {
        int __offset = __offset(26);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public VertexAttribute blendAttributes(int i) {
        return blendAttributes(new VertexAttribute(), i);
    }

    public int blendAttributesLength() {
        int __offset = __offset(24);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public BlendShape blendShapes(int i) {
        return blendShapes(new BlendShape(), i);
    }

    public int blendShapesLength() {
        int __offset = __offset(22);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int indices16(int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.getShort((i * 2) + __vector(__offset)) & 65535;
        }
        return 0;
    }

    public ByteBuffer indices16AsByteBuffer() {
        return __vector_as_bytebuffer(6, 2);
    }

    public ByteBuffer indices16InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 2);
    }

    public int indices16Length() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public long indices32(int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return this.bb.getInt((i * 4) + __vector(__offset)) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public ByteBuffer indices32AsByteBuffer() {
        return __vector_as_bytebuffer(8, 4);
    }

    public ByteBuffer indices32InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 8, 4);
    }

    public int indices32Length() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public boolean interleaved() {
        int __offset = __offset(18);
        return __offset == 0 || this.bb.get(__offset + this.bb_pos) != 0;
    }

    public MaterialDef materials(int i) {
        return materials(new MaterialDef(), i);
    }

    public int materialsLength() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public long numVertices() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public ModelIndexRange ranges(int i) {
        return ranges(new ModelIndexRange(), i);
    }

    public int rangesLength() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int shaderToMeshBones(int i) {
        int __offset = __offset(20);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer shaderToMeshBonesAsByteBuffer() {
        return __vector_as_bytebuffer(20, 1);
    }

    public ByteBuffer shaderToMeshBonesInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 20, 1);
    }

    public int shaderToMeshBonesLength() {
        int __offset = __offset(20);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public VertexAttribute vertexAttributes(int i) {
        return vertexAttributes(new VertexAttribute(), i);
    }

    public int vertexAttributesLength() {
        int __offset = __offset(14);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int vertexData(int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer vertexDataAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer vertexDataInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public int vertexDataLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static int createShaderToMeshBonesVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static int createVertexDataVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static ModelInstanceDef getRootAsModelInstanceDef(ByteBuffer byteBuffer, ModelInstanceDef modelInstanceDef) {
        return modelInstanceDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public SubmeshAabb aabbs(SubmeshAabb submeshAabb, int i) {
        int __offset = __offset(26);
        if (__offset != 0) {
            return submeshAabb.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public VertexAttribute blendAttributes(VertexAttribute vertexAttribute, int i) {
        int __offset = __offset(24);
        if (__offset != 0) {
            return vertexAttribute.__assign((i * 8) + __vector(__offset), this.bb);
        }
        return null;
    }

    public BlendShape blendShapes(BlendShape blendShape, int i) {
        int __offset = __offset(22);
        if (__offset != 0) {
            return blendShape.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public MaterialDef materials(MaterialDef materialDef, int i) {
        int __offset = __offset(12);
        if (__offset != 0) {
            return materialDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public ModelIndexRange ranges(ModelIndexRange modelIndexRange, int i) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return modelIndexRange.__assign((i * 8) + __vector(__offset), this.bb);
        }
        return null;
    }

    public VertexAttribute vertexAttributes(VertexAttribute vertexAttribute, int i) {
        int __offset = __offset(14);
        if (__offset != 0) {
            return vertexAttribute.__assign((i * 8) + __vector(__offset), this.bb);
        }
        return null;
    }
}