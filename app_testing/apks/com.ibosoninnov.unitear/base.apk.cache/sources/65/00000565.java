package b.t;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/* compiled from: ViewModel.java */
/* loaded from: classes.dex */
public abstract class s {

    /* renamed from: a  reason: collision with root package name */
    public final Map<String, Object> f2600a = new HashMap();

    /* renamed from: b  reason: collision with root package name */
    public volatile boolean f2601b = false;

    public void a() {
    }

    public <T> T b(String str, T t) {
        Object obj;
        synchronized (this.f2600a) {
            obj = this.f2600a.get(str);
            if (obj == null) {
                this.f2600a.put(str, t);
            }
        }
        if (obj != null) {
            t = obj;
        }
        if (this.f2601b && (t instanceof Closeable)) {
            try {
                ((Closeable) t).close();
            } catch (IOException e2) {
                throw new RuntimeException(e2);
            }
        }
        return t;
    }
}