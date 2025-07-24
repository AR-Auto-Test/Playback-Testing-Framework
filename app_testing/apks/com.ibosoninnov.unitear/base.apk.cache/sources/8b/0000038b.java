package b.d.b;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/* compiled from: ForwardingImageProxy.java */
/* loaded from: classes.dex */
public abstract class p0 implements r0 {

    /* renamed from: b  reason: collision with root package name */
    public final r0 f1662b;

    /* renamed from: c  reason: collision with root package name */
    public final Set<a> f1663c = new HashSet();

    /* compiled from: ForwardingImageProxy.java */
    /* loaded from: classes.dex */
    public interface a {
        void b(r0 r0Var);
    }

    public p0(r0 r0Var) {
        this.f1662b = r0Var;
    }

    @Override // b.d.b.r0, java.lang.AutoCloseable
    public void close() {
        HashSet hashSet;
        synchronized (this) {
            this.f1662b.close();
        }
        synchronized (this) {
            hashSet = new HashSet(this.f1663c);
        }
        Iterator it = hashSet.iterator();
        while (it.hasNext()) {
            ((a) it.next()).b(this);
        }
    }
}