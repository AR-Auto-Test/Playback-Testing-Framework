package c.c.a.m.x.d;

import c.c.a.m.v.w;
import java.util.Objects;

/* compiled from: BytesResource.java */
/* loaded from: classes.dex */
public class b implements w<byte[]> {

    /* renamed from: b  reason: collision with root package name */
    public final byte[] f4021b;

    public b(byte[] bArr) {
        Objects.requireNonNull(bArr, "Argument must not be null");
        this.f4021b = bArr;
    }

    @Override // c.c.a.m.v.w
    public void a() {
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return this.f4021b.length;
    }

    @Override // c.c.a.m.v.w
    public Class<byte[]> d() {
        return byte[].class;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.v.w
    public byte[] get() {
        return this.f4021b;
    }
}