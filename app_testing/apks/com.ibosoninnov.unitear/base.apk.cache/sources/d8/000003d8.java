package b.g.a;

/* compiled from: CallbackToFutureAdapter.java */
/* loaded from: classes.dex */
public final class b<T> {

    /* renamed from: a  reason: collision with root package name */
    public Object f1805a;

    /* renamed from: b  reason: collision with root package name */
    public e<T> f1806b;

    /* renamed from: c  reason: collision with root package name */
    public f<Void> f1807c = new f<>();

    /* renamed from: d  reason: collision with root package name */
    public boolean f1808d;

    public boolean a(T t) {
        boolean z = true;
        this.f1808d = true;
        e<T> eVar = this.f1806b;
        z = (eVar == null || !eVar.f1810c.h(t)) ? false : false;
        if (z) {
            b();
        }
        return z;
    }

    public final void b() {
        this.f1805a = null;
        this.f1806b = null;
        this.f1807c = null;
    }

    public boolean c(Throwable th) {
        boolean z = true;
        this.f1808d = true;
        e<T> eVar = this.f1806b;
        z = (eVar == null || !eVar.f1810c.i(th)) ? false : false;
        if (z) {
            b();
        }
        return z;
    }

    public void finalize() {
        f<Void> fVar;
        e<T> eVar = this.f1806b;
        if (eVar != null && !eVar.isDone()) {
            StringBuilder x = c.b.a.a.a.x("The completer object was garbage collected - this future would otherwise never complete. The tag was: ");
            x.append(this.f1805a);
            eVar.f1810c.i(new c(x.toString()));
        }
        if (this.f1808d || (fVar = this.f1807c) == null) {
            return;
        }
        fVar.h(null);
    }
}