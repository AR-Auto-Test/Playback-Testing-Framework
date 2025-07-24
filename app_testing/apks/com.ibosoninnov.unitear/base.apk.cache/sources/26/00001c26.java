package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;
import java.nio.ByteBuffer;

/* compiled from: ArImage.java */
/* loaded from: classes.dex */
public final class k extends com.google.ar.core.dependencies.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ArImage f5586a;

    /* renamed from: b  reason: collision with root package name */
    private final long f5587b;

    /* renamed from: c  reason: collision with root package name */
    private final int f5588c;

    public k(ArImage arImage, long j, int i) {
        this.f5586a = arImage;
        this.f5587b = j;
        this.f5588c = i;
    }

    @Override // android.media.Image.Plane
    public final ByteBuffer getBuffer() {
        Session session;
        ByteBuffer nativeGetBuffer;
        ArImage arImage = this.f5586a;
        session = arImage.session;
        nativeGetBuffer = arImage.nativeGetBuffer(session.nativeWrapperHandle, this.f5587b, this.f5588c);
        return nativeGetBuffer.asReadOnlyBuffer();
    }

    @Override // android.media.Image.Plane
    public final int getPixelStride() {
        Session session;
        int nativeGetPixelStride;
        ArImage arImage = this.f5586a;
        session = arImage.session;
        nativeGetPixelStride = arImage.nativeGetPixelStride(session.nativeWrapperHandle, this.f5587b, this.f5588c);
        if (nativeGetPixelStride != -1) {
            return nativeGetPixelStride;
        }
        throw new FatalException("Unknown error in ArImage.Plane.getPixelStride().");
    }

    @Override // android.media.Image.Plane
    public final int getRowStride() {
        Session session;
        int nativeGetRowStride;
        ArImage arImage = this.f5586a;
        session = arImage.session;
        nativeGetRowStride = arImage.nativeGetRowStride(session.nativeWrapperHandle, this.f5587b, this.f5588c);
        if (nativeGetRowStride != -1) {
            return nativeGetRowStride;
        }
        throw new FatalException("Unknown error in ArImage.Plane.getRowStride().");
    }
}