package com.google.ar.sceneform.rendering;

import android.media.Image;
import com.google.ar.core.annotations.UsedByReflection;
import java.io.Serializable;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public class EnvironmentalHdrLightEstimate implements Serializable {
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float colorA;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float colorB;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float colorG;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float colorR;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final CubeMapImage[] cubeMap;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float[] direction;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float relativeIntensity;
    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    private final float[] sphericalHarmonics;

    @UsedByReflection("EnvironmentalHdrLightEstimate.java")
    /* loaded from: classes.dex */
    public static class CubeMapImage implements Serializable {
        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        public final int format;
        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        public final int height;
        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        public final CubeMapPlane[] planes;
        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        public final long timestamp;
        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        public final int width;

        @UsedByReflection("EnvironmentalHdrLightEstimate.java")
        /* loaded from: classes.dex */
        public static class CubeMapPlane implements Serializable {
            @UsedByReflection("EnvironmentalHdrLightEstimate.java")
            public final byte[] bytes;
            @UsedByReflection("EnvironmentalHdrLightEstimate.java")
            public final int pixelStride;
            @UsedByReflection("EnvironmentalHdrLightEstimate.java")
            public final int rowStride;

            public CubeMapPlane(Image.Plane plane) {
                ByteBuffer buffer = plane.getBuffer();
                byte[] bArr = new byte[buffer.remaining()];
                this.bytes = bArr;
                buffer.get(bArr);
                this.pixelStride = plane.getPixelStride();
                this.rowStride = plane.getRowStride();
            }
        }

        public CubeMapImage(Image image) {
            this.format = image.getFormat();
            Image.Plane[] planes = image.getPlanes();
            this.planes = new CubeMapPlane[planes.length];
            for (int i = 0; i < planes.length; i++) {
                this.planes[i] = new CubeMapPlane(planes[i]);
            }
            this.height = image.getHeight();
            this.width = image.getWidth();
            this.timestamp = image.getTimestamp();
        }
    }

    public EnvironmentalHdrLightEstimate(float[] fArr, float[] fArr2, Color color, float f2, Image[] imageArr) {
        if (fArr != null) {
            float[] fArr3 = new float[fArr.length];
            this.sphericalHarmonics = fArr3;
            System.arraycopy(fArr, 0, fArr3, 0, fArr.length);
        } else {
            this.sphericalHarmonics = null;
        }
        if (fArr2 != null) {
            float[] fArr4 = new float[fArr2.length];
            this.direction = fArr4;
            System.arraycopy(fArr2, 0, fArr4, 0, fArr2.length);
        } else {
            this.direction = null;
        }
        this.colorR = color.r;
        this.colorG = color.f5628g;
        this.colorB = color.f5627b;
        this.colorA = color.f5626a;
        this.relativeIntensity = f2;
        if (imageArr != null) {
            this.cubeMap = new CubeMapImage[imageArr.length];
            for (int i = 0; i < imageArr.length; i++) {
                this.cubeMap[i] = new CubeMapImage(imageArr[i]);
            }
            return;
        }
        this.cubeMap = null;
    }

    public Color getColor() {
        return new Color(this.colorR, this.colorG, this.colorB, this.colorA);
    }

    public CubeMapImage[] getCubeMap() {
        return this.cubeMap;
    }

    public float[] getDirection() {
        return this.direction;
    }

    public float getRelativeIntensity() {
        return this.relativeIntensity;
    }

    public float[] getSphericalHarmonics() {
        return this.sphericalHarmonics;
    }
}