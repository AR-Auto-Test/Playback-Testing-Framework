package c.c.a.r;

import c.c.a.m.m;
import c.c.a.s.j;
import java.nio.ByteBuffer;
import java.security.MessageDigest;

/* compiled from: AndroidResourceSignature.java */
/* loaded from: classes.dex */
public final class a implements m {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f4167b = 0;

    /* renamed from: c  reason: collision with root package name */
    public final int f4168c;

    /* renamed from: d  reason: collision with root package name */
    public final m f4169d;

    public a(int i, m mVar) {
        this.f4168c = i;
        this.f4169d = mVar;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        this.f4169d.a(messageDigest);
        messageDigest.update(ByteBuffer.allocate(4).putInt(this.f4168c).array());
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof a) {
            a aVar = (a) obj;
            return this.f4168c == aVar.f4168c && this.f4169d.equals(aVar.f4169d);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return j.g(this.f4169d, this.f4168c);
    }
}