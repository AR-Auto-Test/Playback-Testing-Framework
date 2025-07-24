package b.v;

import android.annotation.SuppressLint;
import b.v.q;
import java.util.HashMap;

/* compiled from: NavigatorProvider.java */
@SuppressLint({"TypeParameterUnusedInFormals"})
/* loaded from: classes.dex */
public class r {

    /* renamed from: a  reason: collision with root package name */
    public static final HashMap<Class<?>, String> f2677a = new HashMap<>();

    /* renamed from: b  reason: collision with root package name */
    public final HashMap<String, q<? extends j>> f2678b = new HashMap<>();

    public static String b(Class<? extends q> cls) {
        HashMap<Class<?>, String> hashMap = f2677a;
        String str = hashMap.get(cls);
        if (str == null) {
            q.b bVar = (q.b) cls.getAnnotation(q.b.class);
            str = bVar != null ? bVar.value() : null;
            if (d(str)) {
                hashMap.put(cls, str);
            } else {
                StringBuilder x = c.b.a.a.a.x("No @Navigator.Name annotation found for ");
                x.append(cls.getSimpleName());
                throw new IllegalArgumentException(x.toString());
            }
        }
        return str;
    }

    public static boolean d(String str) {
        return (str == null || str.isEmpty()) ? false : true;
    }

    public final q<? extends j> a(q<? extends j> qVar) {
        String b2 = b(qVar.getClass());
        if (d(b2)) {
            return this.f2678b.put(b2, qVar);
        }
        throw new IllegalArgumentException("navigator name cannot be an empty string");
    }

    public <T extends q<?>> T c(String str) {
        if (d(str)) {
            q<? extends j> qVar = this.f2678b.get(str);
            if (qVar != null) {
                return qVar;
            }
            throw new IllegalStateException(c.b.a.a.a.r("Could not find Navigator with name \"", str, "\". You must call NavController.addNavigator() for each navigation type."));
        }
        throw new IllegalArgumentException("navigator name cannot be an empty string");
    }
}