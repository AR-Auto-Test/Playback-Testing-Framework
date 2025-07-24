package b.d.a.e;

import android.content.Context;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/* compiled from: Camera2CameraFactory.java */
/* loaded from: classes.dex */
public final class p0 implements b.d.b.d1.y {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.b.d1.d0 f1134a;

    /* renamed from: c  reason: collision with root package name */
    public final b.d.a.e.y1.k f1136c;

    /* renamed from: d  reason: collision with root package name */
    public final List<String> f1137d;

    /* renamed from: e  reason: collision with root package name */
    public final Map<String, r0> f1138e = new HashMap();

    /* renamed from: b  reason: collision with root package name */
    public final b.d.b.d1.c0 f1135b = new b.d.b.d1.c0(1);

    public p0(Context context, b.d.b.d1.d0 d0Var, b.d.b.j0 j0Var) {
        this.f1134a = d0Var;
        b.d.a.e.y1.k a2 = b.d.a.e.y1.k.a(context, d0Var.b());
        this.f1136c = a2;
        try {
            ArrayList arrayList = new ArrayList();
            List<String> asList = Arrays.asList(a2.c());
            if (j0Var == null) {
                for (String str : asList) {
                    arrayList.add(str);
                }
            } else {
                String e2 = b.b.a.e(a2, j0Var.c(), asList);
                ArrayList arrayList2 = new ArrayList();
                for (String str2 : asList) {
                    if (!str2.equals(e2)) {
                        arrayList2.add(c(str2));
                    }
                }
                try {
                    Iterator<b.d.b.i0> it = j0Var.b(arrayList2).iterator();
                    while (it.hasNext()) {
                        arrayList.add(((b.d.b.d1.z) it.next()).b());
                    }
                } catch (IllegalArgumentException unused) {
                }
            }
            this.f1137d = arrayList;
        } catch (b.d.a.e.y1.a e3) {
            throw new b.d.b.t0(b.b.a.d(e3));
        } catch (b.d.b.k0 e4) {
            throw new b.d.b.t0(e4);
        }
    }

    public Set<String> a() {
        return new LinkedHashSet(this.f1137d);
    }

    public b.d.b.d1.a0 b(String str) {
        if (this.f1137d.contains(str)) {
            return new q0(this.f1136c, str, c(str), this.f1135b, this.f1134a.a(), this.f1134a.b());
        }
        throw new IllegalArgumentException("The given camera id is not on the available camera id list.");
    }

    public r0 c(String str) {
        try {
            r0 r0Var = this.f1138e.get(str);
            if (r0Var == null) {
                r0 r0Var2 = new r0(str, this.f1136c.b(str));
                this.f1138e.put(str, r0Var2);
                return r0Var2;
            }
            return r0Var;
        } catch (b.d.a.e.y1.a e2) {
            throw b.b.a.d(e2);
        }
    }
}