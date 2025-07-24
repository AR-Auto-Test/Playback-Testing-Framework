package c.c.a.m.w;

import android.text.TextUtils;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* compiled from: LazyHeaders.java */
/* loaded from: classes.dex */
public final class j implements h {

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, List<i>> f3847b;

    /* renamed from: c  reason: collision with root package name */
    public volatile Map<String, String> f3848c;

    /* compiled from: LazyHeaders.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public static final String f3849a;

        /* renamed from: b  reason: collision with root package name */
        public static final Map<String, List<i>> f3850b;

        /* renamed from: c  reason: collision with root package name */
        public Map<String, List<i>> f3851c = f3850b;

        static {
            String property = System.getProperty("http.agent");
            if (!TextUtils.isEmpty(property)) {
                int length = property.length();
                StringBuilder sb = new StringBuilder(property.length());
                for (int i = 0; i < length; i++) {
                    char charAt = property.charAt(i);
                    if ((charAt > 31 || charAt == '\t') && charAt < 127) {
                        sb.append(charAt);
                    } else {
                        sb.append('?');
                    }
                }
                property = sb.toString();
            }
            f3849a = property;
            HashMap hashMap = new HashMap(2);
            if (!TextUtils.isEmpty(property)) {
                hashMap.put("User-Agent", Collections.singletonList(new b(property)));
            }
            f3850b = Collections.unmodifiableMap(hashMap);
        }
    }

    /* compiled from: LazyHeaders.java */
    /* loaded from: classes.dex */
    public static final class b implements i {

        /* renamed from: a  reason: collision with root package name */
        public final String f3852a;

        public b(String str) {
            this.f3852a = str;
        }

        @Override // c.c.a.m.w.i
        public String a() {
            return this.f3852a;
        }

        public boolean equals(Object obj) {
            if (obj instanceof b) {
                return this.f3852a.equals(((b) obj).f3852a);
            }
            return false;
        }

        public int hashCode() {
            return this.f3852a.hashCode();
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("StringHeaderFactory{value='");
            x.append(this.f3852a);
            x.append('\'');
            x.append('}');
            return x.toString();
        }
    }

    public j(Map<String, List<i>> map) {
        this.f3847b = Collections.unmodifiableMap(map);
    }

    @Override // c.c.a.m.w.h
    public Map<String, String> a() {
        if (this.f3848c == null) {
            synchronized (this) {
                if (this.f3848c == null) {
                    this.f3848c = Collections.unmodifiableMap(b());
                }
            }
        }
        return this.f3848c;
    }

    public final Map<String, String> b() {
        HashMap hashMap = new HashMap();
        for (Map.Entry<String, List<i>> entry : this.f3847b.entrySet()) {
            List<i> value = entry.getValue();
            StringBuilder sb = new StringBuilder();
            int size = value.size();
            for (int i = 0; i < size; i++) {
                String a2 = value.get(i).a();
                if (!TextUtils.isEmpty(a2)) {
                    sb.append(a2);
                    if (i != value.size() - 1) {
                        sb.append(',');
                    }
                }
            }
            String sb2 = sb.toString();
            if (!TextUtils.isEmpty(sb2)) {
                hashMap.put(entry.getKey(), sb2);
            }
        }
        return hashMap;
    }

    public boolean equals(Object obj) {
        if (obj instanceof j) {
            return this.f3847b.equals(((j) obj).f3847b);
        }
        return false;
    }

    public int hashCode() {
        return this.f3847b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("LazyHeaders{headers=");
        x.append(this.f3847b);
        x.append('}');
        return x.toString();
    }
}