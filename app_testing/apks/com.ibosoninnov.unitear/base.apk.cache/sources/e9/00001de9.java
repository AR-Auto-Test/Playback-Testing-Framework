package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class MaterialDef extends Table {
    public static void addCompiledIndex(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addByte(1, (byte) i, 0);
    }

    public static void addName(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addParameters(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createMaterialDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3) {
        flatBufferBuilder.startObject(3);
        addName(flatBufferBuilder, i3);
        addParameters(flatBufferBuilder, i);
        addCompiledIndex(flatBufferBuilder, i2);
        return endMaterialDef(flatBufferBuilder);
    }

    public static int createParametersVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static MaterialDef getRootAsMaterialDef(ByteBuffer byteBuffer) {
        return getRootAsMaterialDef(byteBuffer, new MaterialDef());
    }

    public static void startMaterialDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(3);
    }

    public static void startParametersVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public MaterialDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int compiledIndex() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public String name() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer nameAsByteBuffer() {
        return __vector_as_bytebuffer(8, 1);
    }

    public ByteBuffer nameInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 8, 1);
    }

    public ParameterDef parameters(int i) {
        return parameters(new ParameterDef(), i);
    }

    public int parametersLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static MaterialDef getRootAsMaterialDef(ByteBuffer byteBuffer, MaterialDef materialDef) {
        return materialDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public ParameterDef parameters(ParameterDef parameterDef, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return parameterDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}