package com.google.android.filament;

import java.nio.Buffer;
import java.nio.BufferOverflowException;

/* loaded from: classes.dex */
public class VertexBuffer {
    private long mNativeObject;

    /* loaded from: classes.dex */
    public enum AttributeType {
        BYTE,
        BYTE2,
        BYTE3,
        BYTE4,
        UBYTE,
        UBYTE2,
        UBYTE3,
        UBYTE4,
        SHORT,
        SHORT2,
        SHORT3,
        SHORT4,
        USHORT,
        USHORT2,
        USHORT3,
        USHORT4,
        INT,
        UINT,
        FLOAT,
        FLOAT2,
        FLOAT3,
        FLOAT4,
        HALF,
        HALF2,
        HALF3,
        HALF4
    }

    /* loaded from: classes.dex */
    public static class Builder {
        private final BuilderFinalizer mFinalizer;
        private final long mNativeBuilder;

        /* loaded from: classes.dex */
        public static class BuilderFinalizer {
            private final long mNativeObject;

            public BuilderFinalizer(long j) {
                this.mNativeObject = j;
            }

            public void finalize() {
                try {
                    super.finalize();
                } catch (Throwable unused) {
                }
                VertexBuffer.nDestroyBuilder(this.mNativeObject);
            }
        }

        public Builder() {
            long nCreateBuilder = VertexBuffer.nCreateBuilder();
            this.mNativeBuilder = nCreateBuilder;
            this.mFinalizer = new BuilderFinalizer(nCreateBuilder);
        }

        public Builder attribute(VertexAttribute vertexAttribute, int i, AttributeType attributeType, int i2, int i3) {
            VertexBuffer.nBuilderAttribute(this.mNativeBuilder, vertexAttribute.ordinal(), i, attributeType.ordinal(), i2, i3);
            return this;
        }

        public Builder bufferCount(int i) {
            VertexBuffer.nBuilderBufferCount(this.mNativeBuilder, i);
            return this;
        }

        public VertexBuffer build(Engine engine) {
            long nBuilderBuild = VertexBuffer.nBuilderBuild(this.mNativeBuilder, engine.getNativeObject());
            if (nBuilderBuild != 0) {
                return new VertexBuffer(nBuilderBuild);
            }
            throw new IllegalStateException("Couldn't create VertexBuffer");
        }

        public Builder normalized(VertexAttribute vertexAttribute) {
            VertexBuffer.nBuilderNormalized(this.mNativeBuilder, vertexAttribute.ordinal(), true);
            return this;
        }

        public Builder vertexCount(int i) {
            VertexBuffer.nBuilderVertexCount(this.mNativeBuilder, i);
            return this;
        }

        public Builder normalized(VertexAttribute vertexAttribute, boolean z) {
            VertexBuffer.nBuilderNormalized(this.mNativeBuilder, vertexAttribute.ordinal(), z);
            return this;
        }

        public Builder attribute(VertexAttribute vertexAttribute, int i, AttributeType attributeType) {
            return attribute(vertexAttribute, i, attributeType, 0, 0);
        }
    }

    /* loaded from: classes.dex */
    public static class QuatTangentContext {
        public Buffer normals;
        public int normalsStride;
        public Buffer outBuffer;
        public int outStride;
        public int quatCount;
        public QuatType quatType;
        public Buffer tangents;
        public int tangentsStride;
    }

    /* loaded from: classes.dex */
    public enum QuatType {
        HALF4,
        SHORT4,
        FLOAT4
    }

    /* loaded from: classes.dex */
    public enum VertexAttribute {
        POSITION,
        TANGENTS,
        COLOR,
        UV0,
        UV1,
        BONE_INDICES,
        BONE_WEIGHTS,
        UNUSED,
        CUSTOM0,
        CUSTOM1,
        CUSTOM2,
        CUSTOM3,
        CUSTOM4,
        CUSTOM5,
        CUSTOM6,
        CUSTOM7
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderAttribute(long j, int i, int i2, int i3, int i4, int i5);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderBufferCount(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderNormalized(long j, int i, boolean z);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderVertexCount(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder();

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native int nGetVertexCount(long j);

    private static native void nPopulateTangentQuaternions(int i, int i2, Buffer buffer, int i3, int i4, Buffer buffer2, int i5, int i6, Buffer buffer3, int i7, int i8);

    private static native int nSetBufferAt(long j, long j2, int i, Buffer buffer, int i2, int i3, int i4, Object obj, Runnable runnable);

    public static void populateTangentQuaternions(QuatTangentContext quatTangentContext) {
        int ordinal = quatTangentContext.quatType.ordinal();
        int i = quatTangentContext.quatCount;
        Buffer buffer = quatTangentContext.outBuffer;
        int remaining = buffer.remaining();
        int i2 = quatTangentContext.outStride;
        Buffer buffer2 = quatTangentContext.normals;
        int remaining2 = buffer2.remaining();
        int i3 = quatTangentContext.normalsStride;
        Buffer buffer3 = quatTangentContext.tangents;
        nPopulateTangentQuaternions(ordinal, i, buffer, remaining, i2, buffer2, remaining2, i3, buffer3, buffer3 != null ? buffer3.remaining() : 0, quatTangentContext.tangentsStride);
    }

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed VertexBuffer");
    }

    public int getVertexCount() {
        return nGetVertexCount(getNativeObject());
    }

    public void setBufferAt(Engine engine, int i, Buffer buffer) {
        setBufferAt(engine, i, buffer, 0, 0, null, null);
    }

    private VertexBuffer(long j) {
        this.mNativeObject = j;
    }

    public void setBufferAt(Engine engine, int i, Buffer buffer, int i2, int i3) {
        setBufferAt(engine, i, buffer, i2, i3, null, null);
    }

    public void setBufferAt(Engine engine, int i, Buffer buffer, int i2, int i3, Object obj, Runnable runnable) {
        if (nSetBufferAt(getNativeObject(), engine.getNativeObject(), i, buffer, buffer.remaining(), i2, i3 == 0 ? buffer.remaining() : i3, obj, runnable) < 0) {
            throw new BufferOverflowException();
        }
    }
}