package b.t;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/* compiled from: ViewModelStore.java */
/* loaded from: classes.dex */
public class y {

    /* renamed from: a  reason: collision with root package name */
    public final HashMap<String, s> f2604a = new HashMap<>();

    public final void a() {
        for (s sVar : this.f2604a.values()) {
            sVar.f2601b = true;
            Map<String, Object> map = sVar.f2600a;
            if (map != null) {
                synchronized (map) {
                    for (Object obj : sVar.f2600a.values()) {
                        if (obj instanceof Closeable) {
                            try {
                                ((Closeable) obj).close();
                            } catch (IOException e2) {
                                throw new RuntimeException(e2);
                            }
                        }
                    }
                }
            }
            sVar.a();
        }
        this.f2604a.clear();
    }
}