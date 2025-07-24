package c.c.a.n;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;
import java.util.WeakHashMap;

/* compiled from: ActivityFragmentLifecycle.java */
/* loaded from: classes.dex */
public class a implements l {

    /* renamed from: a  reason: collision with root package name */
    public final Set<m> f4075a = Collections.newSetFromMap(new WeakHashMap());

    /* renamed from: b  reason: collision with root package name */
    public boolean f4076b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f4077c;

    @Override // c.c.a.n.l
    public void a(m mVar) {
        this.f4075a.add(mVar);
        if (this.f4077c) {
            mVar.onDestroy();
        } else if (this.f4076b) {
            mVar.onStart();
        } else {
            mVar.onStop();
        }
    }

    @Override // c.c.a.n.l
    public void b(m mVar) {
        this.f4075a.remove(mVar);
    }

    public void c() {
        this.f4077c = true;
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4075a)).iterator();
        while (it.hasNext()) {
            ((m) it.next()).onDestroy();
        }
    }

    public void d() {
        this.f4076b = true;
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4075a)).iterator();
        while (it.hasNext()) {
            ((m) it.next()).onStart();
        }
    }

    public void e() {
        this.f4076b = false;
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4075a)).iterator();
        while (it.hasNext()) {
            ((m) it.next()).onStop();
        }
    }
}