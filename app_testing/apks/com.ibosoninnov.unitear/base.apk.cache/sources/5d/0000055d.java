package b.t;

import androidx.lifecycle.LiveData;

/* compiled from: MutableLiveData.java */
/* loaded from: classes.dex */
public class m<T> extends LiveData<T> {
    public m(T t) {
        super(t);
    }

    @Override // androidx.lifecycle.LiveData
    public void h(T t) {
        LiveData.a("setValue");
        this.f317g++;
        this.f315e = t;
        c(null);
    }

    public void i(T t) {
        boolean z;
        synchronized (this.f312b) {
            z = this.f316f == LiveData.f311a;
            this.f316f = t;
        }
        if (z) {
            b.c.a.a.a.c().f985b.b(this.j);
        }
    }

    public m() {
    }
}