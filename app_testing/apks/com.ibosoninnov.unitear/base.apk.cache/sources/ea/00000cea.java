package com.google.android.filament;

import java.nio.Buffer;

/* loaded from: classes.dex */
public class SurfaceOrientation {
    private long mNativeObject;

    /* loaded from: classes.dex */
    public static class Builder {
        private Buffer mNormals;
        private int mNormalsStride;
        private Buffer mPositions;
        private int mPositionsStride;
        private Buffer mTangents;
        private int mTangentsStride;
        private Buffer mTexCoords;
        private int mTexCoordsStride;
        private int mTriangleCount;
        private Buffer mTrianglesUint16;
        private Buffer mTrianglesUint32;
        private int mVertexCount;

        public SurfaceOrientation build() {
            long nCreateBuilder = SurfaceOrientation.nCreateBuilder();
            SurfaceOrientation.nBuilderVertexCount(nCreateBuilder, this.mVertexCount);
            SurfaceOrientation.nBuilderTriangleCount(nCreateBuilder, this.mTriangleCount);
            Buffer buffer = this.mNormals;
            if (buffer != null) {
                SurfaceOrientation.nBuilderNormals(nCreateBuilder, buffer, buffer.remaining(), this.mNormalsStride);
            }
            Buffer buffer2 = this.mTangents;
            if (buffer2 != null) {
                SurfaceOrientation.nBuilderTangents(nCreateBuilder, buffer2, buffer2.remaining(), this.mTangentsStride);
            }
            Buffer buffer3 = this.mTexCoords;
            if (buffer3 != null) {
                SurfaceOrientation.nBuilderUVs(nCreateBuilder, buffer3, buffer3.remaining(), this.mTexCoordsStride);
            }
            Buffer buffer4 = this.mPositions;
            if (buffer4 != null) {
                SurfaceOrientation.nBuilderPositions(nCreateBuilder, buffer4, buffer4.remaining(), this.mPositionsStride);
            }
            Buffer buffer5 = this.mTrianglesUint16;
            if (buffer5 != null) {
                SurfaceOrientation.nBuilderTriangles16(nCreateBuilder, buffer5, buffer5.remaining());
            }
            Buffer buffer6 = this.mTrianglesUint32;
            if (buffer6 != null) {
                SurfaceOrientation.nBuilderTriangles32(nCreateBuilder, buffer6, buffer6.remaining());
            }
            long nBuilderBuild = SurfaceOrientation.nBuilderBuild(nCreateBuilder);
            SurfaceOrientation.nDestroyBuilder(nCreateBuilder);
            if (nBuilderBuild != 0) {
                return new SurfaceOrientation(nBuilderBuild);
            }
            throw new IllegalStateException("Could not create SurfaceOrientation");
        }

        public Builder normals(Buffer buffer) {
            this.mNormals = buffer;
            this.mNormalsStride = 0;
            return this;
        }

        public Builder positions(Buffer buffer) {
            this.mPositions = buffer;
            this.mPositionsStride = 0;
            return this;
        }

        public Builder tangents(Buffer buffer) {
            this.mTangents = buffer;
            this.mTangentsStride = 0;
            return this;
        }

        public Builder triangleCount(int i) {
            this.mTriangleCount = i;
            return this;
        }

        public Builder triangles_uint16(Buffer buffer) {
            this.mTrianglesUint16 = buffer;
            return this;
        }

        public Builder triangles_uint32(Buffer buffer) {
            this.mTrianglesUint32 = buffer;
            return this;
        }

        public Builder uvs(Buffer buffer) {
            this.mTexCoords = buffer;
            this.mTexCoordsStride = 0;
            return this;
        }

        public Builder vertexCount(int i) {
            this.mVertexCount = i;
            return this;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderNormals(long j, Buffer buffer, int i, int i2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderPositions(long j, Buffer buffer, int i, int i2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderTangents(long j, Buffer buffer, int i, int i2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderTriangleCount(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderTriangles16(long j, Buffer buffer, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderTriangles32(long j, Buffer buffer, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderUVs(long j, Buffer buffer, int i, int i2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderVertexCount(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder();

    private static native void nDestroy(long j);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native void nGetQuatsAsFloat(long j, Buffer buffer, int i);

    private static native void nGetQuatsAsHalf(long j, Buffer buffer, int i);

    private static native void nGetQuatsAsShort(long j, Buffer buffer, int i);

    private static native int nGetVertexCount(long j);

    public void destroy() {
        nDestroy(this.mNativeObject);
        this.mNativeObject = 0L;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed SurfaceOrientation");
    }

    public void getQuatsAsFloat(Buffer buffer) {
        nGetQuatsAsFloat(this.mNativeObject, buffer, buffer.remaining());
    }

    public void getQuatsAsHalf(Buffer buffer) {
        nGetQuatsAsHalf(this.mNativeObject, buffer, buffer.remaining());
    }

    public void getQuatsAsShort(Buffer buffer) {
        nGetQuatsAsShort(this.mNativeObject, buffer, buffer.remaining());
    }

    public int getVertexCount() {
        return nGetVertexCount(this.mNativeObject);
    }

    private SurfaceOrientation(long j) {
        this.mNativeObject = j;
    }
}