package b.v;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import b.v.j;
import java.util.Iterator;
import java.util.NoSuchElementException;

/* compiled from: NavGraph.java */
/* loaded from: classes.dex */
public class k extends j implements Iterable<j> {
    public final b.f.i<j> j;
    public int k;
    public String l;

    /* compiled from: NavGraph.java */
    /* loaded from: classes.dex */
    public class a implements Iterator<j> {

        /* renamed from: b  reason: collision with root package name */
        public int f2655b = -1;

        /* renamed from: c  reason: collision with root package name */
        public boolean f2656c = false;

        public a() {
        }

        @Override // java.util.Iterator
        public boolean hasNext() {
            return this.f2655b + 1 < k.this.j.i();
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // java.util.Iterator
        public j next() {
            if (hasNext()) {
                this.f2656c = true;
                b.f.i<j> iVar = k.this.j;
                int i = this.f2655b + 1;
                this.f2655b = i;
                return iVar.j(i);
            }
            throw new NoSuchElementException();
        }

        @Override // java.util.Iterator
        public void remove() {
            if (this.f2656c) {
                k.this.j.j(this.f2655b).f2644c = null;
                b.f.i<j> iVar = k.this.j;
                int i = this.f2655b;
                Object[] objArr = iVar.f1779e;
                Object obj = objArr[i];
                Object obj2 = b.f.i.f1776b;
                if (obj != obj2) {
                    objArr[i] = obj2;
                    iVar.f1777c = true;
                }
                this.f2655b = i - 1;
                this.f2656c = false;
                return;
            }
            throw new IllegalStateException("You must call next() before you can remove an element");
        }
    }

    public k(q<? extends k> qVar) {
        super(qVar);
        this.j = new b.f.i<>(10);
    }

    @Override // b.v.j
    public j.a c(i iVar) {
        j.a c2 = super.c(iVar);
        a aVar = new a();
        while (aVar.hasNext()) {
            j.a c3 = ((j) aVar.next()).c(iVar);
            if (c3 != null && (c2 == null || c3.compareTo(c2) > 0)) {
                c2 = c3;
            }
        }
        return c2;
    }

    @Override // b.v.j
    public void d(Context context, AttributeSet attributeSet) {
        super.d(context, attributeSet);
        TypedArray obtainAttributes = context.getResources().obtainAttributes(attributeSet, b.v.t.a.f2685d);
        int resourceId = obtainAttributes.getResourceId(0, 0);
        if (resourceId != this.f2645d) {
            this.k = resourceId;
            this.l = null;
            this.l = j.b(context, resourceId);
            obtainAttributes.recycle();
            return;
        }
        throw new IllegalArgumentException("Start destination " + resourceId + " cannot use the same id as the graph " + this);
    }

    public final void e(j jVar) {
        int i = jVar.f2645d;
        if (i != 0) {
            if (i != this.f2645d) {
                j d2 = this.j.d(i);
                if (d2 == jVar) {
                    return;
                }
                if (jVar.f2644c == null) {
                    if (d2 != null) {
                        d2.f2644c = null;
                    }
                    jVar.f2644c = this;
                    this.j.g(jVar.f2645d, jVar);
                    return;
                }
                throw new IllegalStateException("Destination already has a parent set. Call NavGraph.remove() to remove the previous parent.");
            }
            throw new IllegalArgumentException("Destination " + jVar + " cannot have the same id as graph " + this);
        }
        throw new IllegalArgumentException("Destinations must have an id. Call setId() or include an android:id in your navigation XML.");
    }

    public final j f(int i) {
        return g(i, true);
    }

    public final j g(int i, boolean z) {
        k kVar;
        j e2 = this.j.e(i, null);
        if (e2 != null) {
            return e2;
        }
        if (!z || (kVar = this.f2644c) == null) {
            return null;
        }
        return kVar.f(i);
    }

    @Override // java.lang.Iterable
    public final Iterator<j> iterator() {
        return new a();
    }

    @Override // b.v.j
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append(" startDestination=");
        j f2 = f(this.k);
        if (f2 == null) {
            String str = this.l;
            if (str == null) {
                sb.append("0x");
                sb.append(Integer.toHexString(this.k));
            } else {
                sb.append(str);
            }
        } else {
            sb.append("{");
            sb.append(f2.toString());
            sb.append("}");
        }
        return sb.toString();
    }
}