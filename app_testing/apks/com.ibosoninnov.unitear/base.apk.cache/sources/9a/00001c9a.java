package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelPipelineImportDef extends Table {
    public static void addAxisSystem(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addInt(4, i, -1);
    }

    public static void addCmPerUnit(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(13, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addEnsureVertexOrientationWNotZero(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(12, z, false);
    }

    public static void addFile(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addFixInfacingNormals(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(11, z, true);
    }

    public static void addFlattenHierarchyAndTransformVerticesToRootSpace(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(9, z, false);
    }

    public static void addFlipTextureCoordinates(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(8, z, false);
    }

    public static void addMaxBoneWeights(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addInt(6, i, 4);
    }

    public static void addMergeMaterials(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(15, z, true);
    }

    public static void addName(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addRecenter(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(2, z, false);
    }

    public static void addReportErrorsToStdout(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(7, z, false);
    }

    public static void addScale(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(3, f2, 1.0d);
    }

    public static void addSmoothingAngle(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(5, f2, 45.0d);
    }

    public static void addTargetMeshes(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(14, i, 0);
    }

    public static void addUseSpecularGlossinessTexturesIfPresent(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(10, z, false);
    }

    public static int createModelPipelineImportDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, boolean z, float f2, int i3, float f3, int i4, boolean z2, boolean z3, boolean z4, boolean z5, boolean z6, boolean z7, float f4, int i5, boolean z8) {
        flatBufferBuilder.startObject(16);
        addTargetMeshes(flatBufferBuilder, i5);
        addCmPerUnit(flatBufferBuilder, f4);
        addMaxBoneWeights(flatBufferBuilder, i4);
        addSmoothingAngle(flatBufferBuilder, f3);
        addAxisSystem(flatBufferBuilder, i3);
        addScale(flatBufferBuilder, f2);
        addFile(flatBufferBuilder, i2);
        addName(flatBufferBuilder, i);
        addMergeMaterials(flatBufferBuilder, z8);
        addEnsureVertexOrientationWNotZero(flatBufferBuilder, z7);
        addFixInfacingNormals(flatBufferBuilder, z6);
        addUseSpecularGlossinessTexturesIfPresent(flatBufferBuilder, z5);
        addFlattenHierarchyAndTransformVerticesToRootSpace(flatBufferBuilder, z4);
        addFlipTextureCoordinates(flatBufferBuilder, z3);
        addReportErrorsToStdout(flatBufferBuilder, z2);
        addRecenter(flatBufferBuilder, z);
        return endModelPipelineImportDef(flatBufferBuilder);
    }

    public static int createTargetMeshesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endModelPipelineImportDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelPipelineImportDef getRootAsModelPipelineImportDef(ByteBuffer byteBuffer) {
        return getRootAsModelPipelineImportDef(byteBuffer, new ModelPipelineImportDef());
    }

    public static void startModelPipelineImportDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(16);
    }

    public static void startTargetMeshesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public ModelPipelineImportDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int axisSystem() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos);
        }
        return -1;
    }

    public float cmPerUnit() {
        int __offset = __offset(30);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public boolean ensureVertexOrientationWNotZero() {
        int __offset = __offset(28);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public String file() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer fileAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer fileInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 1);
    }

    public boolean fixInfacingNormals() {
        int __offset = __offset(26);
        return __offset == 0 || this.bb.get(__offset + this.bb_pos) != 0;
    }

    public boolean flattenHierarchyAndTransformVerticesToRootSpace() {
        int __offset = __offset(22);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public boolean flipTextureCoordinates() {
        int __offset = __offset(20);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public int maxBoneWeights() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos);
        }
        return 4;
    }

    public boolean mergeMaterials() {
        int __offset = __offset(34);
        return __offset == 0 || this.bb.get(__offset + this.bb_pos) != 0;
    }

    public String name() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer nameAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer nameInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public boolean recenter() {
        int __offset = __offset(8);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public boolean reportErrorsToStdout() {
        int __offset = __offset(18);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public float scale() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return this.bb.getFloat(__offset + this.bb_pos);
        }
        return 1.0f;
    }

    public float smoothingAngle() {
        int __offset = __offset(14);
        if (__offset != 0) {
            return this.bb.getFloat(__offset + this.bb_pos);
        }
        return 45.0f;
    }

    public String targetMeshes(int i) {
        int __offset = __offset(32);
        if (__offset != 0) {
            return __string((i * 4) + __vector(__offset));
        }
        return null;
    }

    public int targetMeshesLength() {
        int __offset = __offset(32);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public boolean useSpecularGlossinessTexturesIfPresent() {
        int __offset = __offset(24);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public static ModelPipelineImportDef getRootAsModelPipelineImportDef(ByteBuffer byteBuffer, ModelPipelineImportDef modelPipelineImportDef) {
        return modelPipelineImportDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}