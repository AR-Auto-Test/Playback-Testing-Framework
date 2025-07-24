package g;

/* compiled from: ForwardingSink.java */
/* loaded from: classes2.dex */
public abstract class i implements w {

    /* renamed from: b  reason: collision with root package name */
    public final w f6183b;

    public i(w wVar) {
        if (wVar != null) {
            this.f6183b = wVar;
            return;
        }
        throw new IllegalArgumentException("delegate == null");
    }

    @Override // g.w
    public y b() {
        return this.f6183b.b();
    }

    @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f6183b.close();
    }

    @Override // g.w, java.io.Flushable
    public void flush() {
        this.f6183b.flush();
    }

    public String toString() {
        return getClass().getSimpleName() + "(" + this.f6183b.toString() + ")";
    }
}