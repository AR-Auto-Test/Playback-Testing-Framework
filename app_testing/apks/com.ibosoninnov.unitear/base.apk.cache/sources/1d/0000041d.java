package b.i.b;

import b.f.h;
import b.j.i.d;
import b.j.i.e;
import java.util.ArrayList;
import java.util.HashSet;

/* compiled from: DirectedAcyclicGraph.java */
/* loaded from: classes.dex */
public final class a<T> {

    /* renamed from: a  reason: collision with root package name */
    public final d<ArrayList<T>> f2019a = new e(10);

    /* renamed from: b  reason: collision with root package name */
    public final h<T, ArrayList<T>> f2020b = new h<>();

    /* renamed from: c  reason: collision with root package name */
    public final ArrayList<T> f2021c = new ArrayList<>();

    /* renamed from: d  reason: collision with root package name */
    public final HashSet<T> f2022d = new HashSet<>();

    public void a(T t) {
        if (this.f2020b.e(t) >= 0) {
            return;
        }
        this.f2020b.put(t, null);
    }

    public final void b(T t, ArrayList<T> arrayList, HashSet<T> hashSet) {
        if (arrayList.contains(t)) {
            return;
        }
        if (!hashSet.contains(t)) {
            hashSet.add(t);
            ArrayList<T> orDefault = this.f2020b.getOrDefault(t, null);
            if (orDefault != null) {
                int size = orDefault.size();
                for (int i = 0; i < size; i++) {
                    b(orDefault.get(i), arrayList, hashSet);
                }
            }
            hashSet.remove(t);
            arrayList.add(t);
            return;
        }
        throw new RuntimeException("This graph contains cyclic dependencies");
    }
}