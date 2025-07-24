package b.d.b.d1;

import android.util.ArrayMap;
import java.util.Map;

/* compiled from: TagBundle.java */
/* loaded from: classes.dex */
public class g1 {

    /* renamed from: a  reason: collision with root package name */
    public static final g1 f1479a = new g1(new ArrayMap());

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, Integer> f1480b;

    public g1(Map<String, Integer> map) {
        this.f1480b = map;
    }

    public Integer a(String str) {
        return this.f1480b.get(str);
    }
}