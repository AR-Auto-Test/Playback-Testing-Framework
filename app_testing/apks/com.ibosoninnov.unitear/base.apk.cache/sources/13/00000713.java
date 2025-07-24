package c.c.a.m;

import android.content.Context;
import c.c.a.m.v.w;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collection;

/* compiled from: MultiTransformation.java */
/* loaded from: classes.dex */
public class n<T> implements t<T> {

    /* renamed from: b  reason: collision with root package name */
    public final Collection<? extends t<T>> f3538b;

    @SafeVarargs
    public n(t<T>... tVarArr) {
        if (tVarArr.length != 0) {
            this.f3538b = Arrays.asList(tVarArr);
            return;
        }
        throw new IllegalArgumentException("MultiTransformation must contain at least one Transformation");
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        for (t<T> tVar : this.f3538b) {
            tVar.a(messageDigest);
        }
    }

    @Override // c.c.a.m.t
    public w<T> b(Context context, w<T> wVar, int i, int i2) {
        w<T> wVar2 = wVar;
        for (t<T> tVar : this.f3538b) {
            w<T> b2 = tVar.b(context, wVar2, i, i2);
            if (wVar2 != null && !wVar2.equals(wVar) && !wVar2.equals(b2)) {
                wVar2.a();
            }
            wVar2 = b2;
        }
        return wVar2;
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof n) {
            return this.f3538b.equals(((n) obj).f3538b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f3538b.hashCode();
    }
}