package c.c.a.m.v.d0;

import c.c.a.m.m;
import c.c.a.s.k.a;
import c.c.a.s.k.d;
import com.google.common.primitives.UnsignedBytes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;

/* compiled from: SafeKeyGenerator.java */
/* loaded from: classes.dex */
public class k {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.s.g<m, String> f3674a = new c.c.a.s.g<>(1000);

    /* renamed from: b  reason: collision with root package name */
    public final b.j.i.d<b> f3675b = c.c.a.s.k.a.a(10, new a(this));

    /* compiled from: SafeKeyGenerator.java */
    /* loaded from: classes.dex */
    public class a implements a.b<b> {
        public a(k kVar) {
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // c.c.a.s.k.a.b
        public b a() {
            try {
                return new b(MessageDigest.getInstance("SHA-256"));
            } catch (NoSuchAlgorithmException e2) {
                throw new RuntimeException(e2);
            }
        }
    }

    /* compiled from: SafeKeyGenerator.java */
    /* loaded from: classes.dex */
    public static final class b implements a.d {

        /* renamed from: b  reason: collision with root package name */
        public final MessageDigest f3676b;

        /* renamed from: c  reason: collision with root package name */
        public final c.c.a.s.k.d f3677c = new d.b();

        public b(MessageDigest messageDigest) {
            this.f3676b = messageDigest;
        }

        @Override // c.c.a.s.k.a.d
        public c.c.a.s.k.d b() {
            return this.f3677c;
        }
    }

    public String a(m mVar) {
        String a2;
        synchronized (this.f3674a) {
            a2 = this.f3674a.a(mVar);
        }
        if (a2 == null) {
            b b2 = this.f3675b.b();
            Objects.requireNonNull(b2, "Argument must not be null");
            b bVar = b2;
            try {
                mVar.a(bVar.f3676b);
                byte[] digest = bVar.f3676b.digest();
                char[] cArr = c.c.a.s.j.f4198b;
                synchronized (cArr) {
                    for (int i = 0; i < digest.length; i++) {
                        int i2 = digest[i] & UnsignedBytes.MAX_VALUE;
                        int i3 = i * 2;
                        char[] cArr2 = c.c.a.s.j.f4197a;
                        cArr[i3] = cArr2[i2 >>> 4];
                        cArr[i3 + 1] = cArr2[i2 & 15];
                    }
                    a2 = new String(cArr);
                }
            } finally {
                this.f3675b.a(bVar);
            }
        }
        synchronized (this.f3674a) {
            this.f3674a.d(mVar, a2);
        }
        return a2;
    }
}