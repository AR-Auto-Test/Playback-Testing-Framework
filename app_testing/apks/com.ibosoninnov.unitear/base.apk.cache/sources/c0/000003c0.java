package b.f;

import java.util.Map;

/* compiled from: ArraySet.java */
/* loaded from: classes.dex */
public class b extends g<E, E> {

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ c f1738d;

    public b(c cVar) {
        this.f1738d = cVar;
    }

    @Override // b.f.g
    public void a() {
        this.f1738d.clear();
    }

    @Override // b.f.g
    public Object b(int i, int i2) {
        return this.f1738d.i[i];
    }

    @Override // b.f.g
    public Map<E, E> c() {
        throw new UnsupportedOperationException("not a map");
    }

    @Override // b.f.g
    public int d() {
        return this.f1738d.j;
    }

    @Override // b.f.g
    public int e(Object obj) {
        return this.f1738d.indexOf(obj);
    }

    @Override // b.f.g
    public int f(Object obj) {
        return this.f1738d.indexOf(obj);
    }

    @Override // b.f.g
    public void g(E e2, E e3) {
        this.f1738d.add(e2);
    }

    @Override // b.f.g
    public void h(int i) {
        this.f1738d.e(i);
    }

    @Override // b.f.g
    public E i(int i, E e2) {
        throw new UnsupportedOperationException("not a map");
    }
}