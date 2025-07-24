package b.t;

import android.os.Bundle;
import android.os.Parcelable;
import b.x.a;
import com.google.firebase.crashlytics.internal.metadata.UserMetadata;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/* compiled from: SavedStateHandle.java */
/* loaded from: classes.dex */
public final class q {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f2589a = 0;

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, Object> f2590b;

    /* renamed from: c  reason: collision with root package name */
    public final a.b f2591c;

    /* compiled from: SavedStateHandle.java */
    /* loaded from: classes.dex */
    public class a implements a.b {
        public a() {
        }

        @Override // b.x.a.b
        public Bundle a() {
            Set<String> keySet = q.this.f2590b.keySet();
            ArrayList<? extends Parcelable> arrayList = new ArrayList<>(keySet.size());
            ArrayList<? extends Parcelable> arrayList2 = new ArrayList<>(arrayList.size());
            for (String str : keySet) {
                arrayList.add(str);
                arrayList2.add(q.this.f2590b.get(str));
            }
            Bundle bundle = new Bundle();
            bundle.putParcelableArrayList(UserMetadata.KEYDATA_FILENAME, arrayList);
            bundle.putParcelableArrayList("values", arrayList2);
            return bundle;
        }
    }

    static {
        Class cls = Double.TYPE;
        Class cls2 = Integer.TYPE;
        Class cls3 = Long.TYPE;
        Class cls4 = Byte.TYPE;
        Class cls5 = Character.TYPE;
        Class cls6 = Float.TYPE;
        Class cls7 = Short.TYPE;
    }

    public q(Map<String, Object> map) {
        new HashMap();
        this.f2591c = new a();
        this.f2590b = new HashMap(map);
    }

    public q() {
        new HashMap();
        this.f2591c = new a();
        this.f2590b = new HashMap();
    }
}