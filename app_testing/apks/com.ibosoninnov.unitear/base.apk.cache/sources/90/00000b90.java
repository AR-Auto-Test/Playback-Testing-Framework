package com.bumptech.glide.load;

import c.c.a.m.v.c0.b;
import java.io.InputStream;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public interface ImageHeaderParser {

    /* loaded from: classes.dex */
    public enum ImageType {
        GIF(true),
        JPEG(false),
        RAW(false),
        PNG_A(true),
        PNG(false),
        WEBP_A(true),
        WEBP(false),
        UNKNOWN(false);
        

        /* renamed from: b  reason: collision with root package name */
        public final boolean f5534b;

        ImageType(boolean z) {
            this.f5534b = z;
        }

        public boolean hasAlpha() {
            return this.f5534b;
        }
    }

    ImageType a(ByteBuffer byteBuffer);

    ImageType b(InputStream inputStream);

    int c(InputStream inputStream, b bVar);
}