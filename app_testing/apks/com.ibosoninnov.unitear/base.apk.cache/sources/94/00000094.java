package androidx.databinding;

import android.util.Log;
import android.view.View;
import b.m.d;
import b.m.e;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;

/* loaded from: classes.dex */
public class MergedDataBinderMapper extends d {

    /* renamed from: a  reason: collision with root package name */
    public Set<Class<? extends d>> f254a = new HashSet();

    /* renamed from: b  reason: collision with root package name */
    public List<d> f255b = new CopyOnWriteArrayList();

    /* renamed from: c  reason: collision with root package name */
    public List<String> f256c = new CopyOnWriteArrayList();

    @Override // b.m.d
    public ViewDataBinding b(e eVar, View view, int i) {
        for (d dVar : this.f255b) {
            ViewDataBinding b2 = dVar.b(eVar, view, i);
            if (b2 != null) {
                return b2;
            }
        }
        if (e()) {
            return b(eVar, view, i);
        }
        return null;
    }

    @Override // b.m.d
    public ViewDataBinding c(e eVar, View[] viewArr, int i) {
        for (d dVar : this.f255b) {
            ViewDataBinding c2 = dVar.c(eVar, viewArr, i);
            if (c2 != null) {
                return c2;
            }
        }
        if (e()) {
            return c(eVar, viewArr, i);
        }
        return null;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: java.util.Set<java.lang.Class<? extends b.m.d>> */
    /* JADX WARN: Multi-variable type inference failed */
    public void d(d dVar) {
        if (this.f254a.add(dVar.getClass())) {
            this.f255b.add(dVar);
            for (d dVar2 : dVar.a()) {
                d(dVar2);
            }
        }
    }

    public final boolean e() {
        boolean z = false;
        for (String str : this.f256c) {
            try {
                Class<?> cls = Class.forName(str);
                if (d.class.isAssignableFrom(cls)) {
                    d((d) cls.newInstance());
                    this.f256c.remove(str);
                    z = true;
                }
            } catch (ClassNotFoundException unused) {
            } catch (IllegalAccessException e2) {
                Log.e("MergedDataBinderMapper", "unable to add feature mapper for " + str, e2);
            } catch (InstantiationException e3) {
                Log.e("MergedDataBinderMapper", "unable to add feature mapper for " + str, e3);
            }
        }
        return z;
    }
}