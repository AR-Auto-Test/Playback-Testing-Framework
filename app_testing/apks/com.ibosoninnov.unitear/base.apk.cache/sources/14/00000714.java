package c.c.a.m;

import android.text.TextUtils;
import java.security.MessageDigest;
import java.util.Objects;

/* compiled from: Option.java */
/* loaded from: classes.dex */
public final class o<T> {

    /* renamed from: a  reason: collision with root package name */
    public static final b<Object> f3539a = new a();

    /* renamed from: b  reason: collision with root package name */
    public final T f3540b;

    /* renamed from: c  reason: collision with root package name */
    public final b<T> f3541c;

    /* renamed from: d  reason: collision with root package name */
    public final String f3542d;

    /* renamed from: e  reason: collision with root package name */
    public volatile byte[] f3543e;

    /* compiled from: Option.java */
    /* loaded from: classes.dex */
    public class a implements b<Object> {
        @Override // c.c.a.m.o.b
        public void a(byte[] bArr, Object obj, MessageDigest messageDigest) {
        }
    }

    /* compiled from: Option.java */
    /* loaded from: classes.dex */
    public interface b<T> {
        void a(byte[] bArr, T t, MessageDigest messageDigest);
    }

    public o(String str, T t, b<T> bVar) {
        if (!TextUtils.isEmpty(str)) {
            this.f3542d = str;
            this.f3540b = t;
            Objects.requireNonNull(bVar, "Argument must not be null");
            this.f3541c = bVar;
            return;
        }
        throw new IllegalArgumentException("Must not be null or empty");
    }

    public static <T> o<T> a(String str, T t) {
        return new o<>(str, t, f3539a);
    }

    public boolean equals(Object obj) {
        if (obj instanceof o) {
            return this.f3542d.equals(((o) obj).f3542d);
        }
        return false;
    }

    public int hashCode() {
        return this.f3542d.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Option{key='");
        x.append(this.f3542d);
        x.append('\'');
        x.append('}');
        return x.toString();
    }
}