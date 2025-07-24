package b.v;

import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.net.Uri;
import android.os.Bundle;
import android.util.AttributeSet;
import b.v.h;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* compiled from: NavDestination.java */
/* loaded from: classes.dex */
public class j {

    /* renamed from: b  reason: collision with root package name */
    public final String f2643b;

    /* renamed from: c  reason: collision with root package name */
    public k f2644c;

    /* renamed from: d  reason: collision with root package name */
    public int f2645d;

    /* renamed from: e  reason: collision with root package name */
    public String f2646e;

    /* renamed from: f  reason: collision with root package name */
    public CharSequence f2647f;

    /* renamed from: g  reason: collision with root package name */
    public ArrayList<h> f2648g;

    /* renamed from: h  reason: collision with root package name */
    public b.f.i<c> f2649h;
    public HashMap<String, d> i;

    /* compiled from: NavDestination.java */
    /* loaded from: classes.dex */
    public static class a implements Comparable<a> {

        /* renamed from: b  reason: collision with root package name */
        public final j f2650b;

        /* renamed from: c  reason: collision with root package name */
        public final Bundle f2651c;

        /* renamed from: d  reason: collision with root package name */
        public final boolean f2652d;

        /* renamed from: e  reason: collision with root package name */
        public final boolean f2653e;

        /* renamed from: f  reason: collision with root package name */
        public final int f2654f;

        public a(j jVar, Bundle bundle, boolean z, boolean z2, int i) {
            this.f2650b = jVar;
            this.f2651c = bundle;
            this.f2652d = z;
            this.f2653e = z2;
            this.f2654f = i;
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // java.lang.Comparable
        /* renamed from: a */
        public int compareTo(a aVar) {
            boolean z = this.f2652d;
            if (!z || aVar.f2652d) {
                if (z || !aVar.f2652d) {
                    Bundle bundle = this.f2651c;
                    if (bundle == null || aVar.f2651c != null) {
                        if (bundle != null || aVar.f2651c == null) {
                            if (bundle != null) {
                                int size = bundle.size() - aVar.f2651c.size();
                                if (size > 0) {
                                    return 1;
                                }
                                if (size < 0) {
                                    return -1;
                                }
                            }
                            boolean z2 = this.f2653e;
                            if (!z2 || aVar.f2653e) {
                                if (z2 || !aVar.f2653e) {
                                    return this.f2654f - aVar.f2654f;
                                }
                                return -1;
                            }
                            return 1;
                        }
                        return -1;
                    }
                    return 1;
                }
                return -1;
            }
            return 1;
        }
    }

    static {
        new HashMap();
    }

    public j(q<? extends j> qVar) {
        this.f2643b = r.b(qVar.getClass());
    }

    public static String b(Context context, int i) {
        if (i <= 16777215) {
            return Integer.toString(i);
        }
        try {
            return context.getResources().getResourceName(i);
        } catch (Resources.NotFoundException unused) {
            return Integer.toString(i);
        }
    }

    public Bundle a(Bundle bundle) {
        HashMap<String, d> hashMap;
        if (bundle == null && ((hashMap = this.i) == null || hashMap.isEmpty())) {
            return null;
        }
        Bundle bundle2 = new Bundle();
        HashMap<String, d> hashMap2 = this.i;
        if (hashMap2 != null) {
            for (Map.Entry<String, d> entry : hashMap2.entrySet()) {
                d value = entry.getValue();
                String key = entry.getKey();
                if (value.f2613c) {
                    value.f2611a.d(bundle2, key, value.f2614d);
                }
            }
        }
        if (bundle != null) {
            bundle2.putAll(bundle);
            HashMap<String, d> hashMap3 = this.i;
            if (hashMap3 != null) {
                for (Map.Entry<String, d> entry2 : hashMap3.entrySet()) {
                    d value2 = entry2.getValue();
                    String key2 = entry2.getKey();
                    boolean z = false;
                    if (value2.f2612b || !bundle2.containsKey(key2) || bundle2.get(key2) != null) {
                        try {
                            value2.f2611a.a(bundle2, key2);
                            z = true;
                            continue;
                        } catch (ClassCastException unused) {
                            continue;
                        }
                    }
                    if (!z) {
                        StringBuilder x = c.b.a.a.a.x("Wrong argument type for '");
                        x.append(entry2.getKey());
                        x.append("' in argument bundle. ");
                        x.append(entry2.getValue().f2611a.b());
                        x.append(" expected.");
                        throw new IllegalArgumentException(x.toString());
                    }
                }
            }
        }
        return bundle2;
    }

    public a c(i iVar) {
        Bundle bundle;
        int i;
        Bundle bundle2;
        Matcher matcher;
        Uri uri;
        ArrayList<h> arrayList = this.f2648g;
        Matcher matcher2 = null;
        if (arrayList == null) {
            return null;
        }
        Iterator<h> it = arrayList.iterator();
        a aVar = null;
        while (it.hasNext()) {
            h next = it.next();
            Uri uri2 = iVar.f2640a;
            if (uri2 != null) {
                HashMap<String, d> hashMap = this.i;
                Map emptyMap = hashMap == null ? Collections.emptyMap() : Collections.unmodifiableMap(hashMap);
                Matcher matcher3 = next.f2631d.matcher(uri2.toString());
                if (matcher3.matches()) {
                    bundle2 = new Bundle();
                    int size = next.f2629b.size();
                    int i2 = 0;
                    while (true) {
                        if (i2 < size) {
                            String str = next.f2629b.get(i2);
                            i2++;
                            if (next.b(bundle2, str, Uri.decode(matcher3.group(i2)), (d) emptyMap.get(str))) {
                                break;
                            }
                        } else if (next.f2633f) {
                            Iterator<String> it2 = next.f2630c.keySet().iterator();
                            while (true) {
                                if (!it2.hasNext()) {
                                    break;
                                }
                                String next2 = it2.next();
                                h.b bVar = next.f2630c.get(next2);
                                String queryParameter = uri2.getQueryParameter(next2);
                                if (queryParameter != null) {
                                    matcher = Pattern.compile(bVar.f2638a).matcher(queryParameter);
                                    if (!matcher.matches()) {
                                        break;
                                    }
                                } else {
                                    matcher = matcher2;
                                }
                                int i3 = 0;
                                while (i3 < bVar.f2639b.size()) {
                                    String decode = matcher != null ? Uri.decode(matcher.group(i3 + 1)) : matcher2;
                                    String str2 = bVar.f2639b.get(i3);
                                    d dVar = (d) emptyMap.get(str2);
                                    if (decode != null) {
                                        uri = uri2;
                                        if (!decode.replaceAll("[{}]", "").equals(str2) && next.b(bundle2, str2, decode, dVar)) {
                                            bundle2 = null;
                                            break;
                                        }
                                    } else {
                                        uri = uri2;
                                    }
                                    i3++;
                                    uri2 = uri;
                                    matcher2 = null;
                                }
                            }
                        }
                    }
                    bundle = bundle2;
                }
                bundle2 = matcher2;
                bundle = bundle2;
            } else {
                bundle = null;
            }
            String str3 = iVar.f2641b;
            boolean z = str3 != null && str3.equals(next.f2634g);
            String str4 = iVar.f2642c;
            if (str4 != null) {
                i = (next.i == null || !next.f2635h.matcher(str4).matches()) ? -1 : new h.a(next.i).compareTo(new h.a(str4));
            } else {
                i = -1;
            }
            if (bundle != null || z || i > -1) {
                a aVar2 = new a(this, bundle, next.f2632e, z, i);
                if (aVar == null || aVar2.compareTo(aVar) > 0) {
                    aVar = aVar2;
                }
            }
            matcher2 = null;
        }
        return aVar;
    }

    public void d(Context context, AttributeSet attributeSet) {
        TypedArray obtainAttributes = context.getResources().obtainAttributes(attributeSet, b.v.t.a.f2686e);
        int resourceId = obtainAttributes.getResourceId(1, 0);
        this.f2645d = resourceId;
        this.f2646e = null;
        this.f2646e = b(context, resourceId);
        this.f2647f = obtainAttributes.getText(0);
        obtainAttributes.recycle();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append("(");
        String str = this.f2646e;
        if (str == null) {
            sb.append("0x");
            sb.append(Integer.toHexString(this.f2645d));
        } else {
            sb.append(str);
        }
        sb.append(")");
        if (this.f2647f != null) {
            sb.append(" label=");
            sb.append(this.f2647f);
        }
        return sb.toString();
    }
}