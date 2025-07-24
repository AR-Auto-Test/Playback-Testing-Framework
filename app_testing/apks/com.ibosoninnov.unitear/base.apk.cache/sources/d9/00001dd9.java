package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class CompiledMaterialDef extends Table {
    public static void addCompiledMaterial(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addCompressedMaterial(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addDecl(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addSha1sum(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static int createCompiledMaterialDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4) {
        flatBufferBuilder.startObject(4);
        addCompressedMaterial(flatBufferBuilder, i4);
        addDecl(flatBufferBuilder, i3);
        addSha1sum(flatBufferBuilder, i2);
        addCompiledMaterial(flatBufferBuilder, i);
        return endCompiledMaterialDef(flatBufferBuilder);
    }

    public static int createCompiledMaterialVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int endCompiledMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static CompiledMaterialDef getRootAsCompiledMaterialDef(ByteBuffer byteBuffer) {
        return getRootAsCompiledMaterialDef(byteBuffer, new CompiledMaterialDef());
    }

    public static void startCompiledMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public static void startCompiledMaterialVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public CompiledMaterialDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int compiledMaterial(int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer compiledMaterialAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer compiledMaterialInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public int compiledMaterialLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public String compressedMaterial() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer compressedMaterialAsByteBuffer() {
        return __vector_as_bytebuffer(10, 1);
    }

    public ByteBuffer compressedMaterialInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 10, 1);
    }

    public CompiledMaterialDeclDef decl() {
        return decl(new CompiledMaterialDeclDef());
    }

    public String sha1sum() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer sha1sumAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer sha1sumInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 1);
    }

    public static int createCompiledMaterialVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static CompiledMaterialDef getRootAsCompiledMaterialDef(ByteBuffer byteBuffer, CompiledMaterialDef compiledMaterialDef) {
        return compiledMaterialDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public CompiledMaterialDeclDef decl(CompiledMaterialDeclDef compiledMaterialDeclDef) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return compiledMaterialDeclDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}