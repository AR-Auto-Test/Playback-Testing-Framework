package b.j.j.x;

import android.os.Build;
import android.os.Bundle;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityNodeProvider;
import java.util.List;
import java.util.Objects;

/* compiled from: AccessibilityNodeProviderCompat.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public final Object f2272a;

    /* compiled from: AccessibilityNodeProviderCompat.java */
    /* loaded from: classes.dex */
    public static class a extends AccessibilityNodeProvider {

        /* renamed from: a  reason: collision with root package name */
        public final c f2273a;

        public a(c cVar) {
            this.f2273a = cVar;
        }

        @Override // android.view.accessibility.AccessibilityNodeProvider
        public AccessibilityNodeInfo createAccessibilityNodeInfo(int i) {
            b.j.j.x.b a2 = this.f2273a.a(i);
            if (a2 == null) {
                return null;
            }
            return a2.f2259b;
        }

        @Override // android.view.accessibility.AccessibilityNodeProvider
        public List<AccessibilityNodeInfo> findAccessibilityNodeInfosByText(String str, int i) {
            Objects.requireNonNull(this.f2273a);
            return null;
        }

        @Override // android.view.accessibility.AccessibilityNodeProvider
        public boolean performAction(int i, int i2, Bundle bundle) {
            return this.f2273a.c(i, i2, bundle);
        }
    }

    /* compiled from: AccessibilityNodeProviderCompat.java */
    /* loaded from: classes.dex */
    public static class b extends a {
        public b(c cVar) {
            super(cVar);
        }

        @Override // android.view.accessibility.AccessibilityNodeProvider
        public AccessibilityNodeInfo findFocus(int i) {
            b.j.j.x.b b2 = this.f2273a.b(i);
            if (b2 == null) {
                return null;
            }
            return b2.f2259b;
        }
    }

    /* compiled from: AccessibilityNodeProviderCompat.java */
    /* renamed from: b.j.j.x.c$c  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0038c extends b {
        public C0038c(c cVar) {
            super(cVar);
        }

        @Override // android.view.accessibility.AccessibilityNodeProvider
        public void addExtraDataToAccessibilityNodeInfo(int i, AccessibilityNodeInfo accessibilityNodeInfo, String str, Bundle bundle) {
            Objects.requireNonNull(this.f2273a);
        }
    }

    public c() {
        if (Build.VERSION.SDK_INT >= 26) {
            this.f2272a = new C0038c(this);
        } else {
            this.f2272a = new b(this);
        }
    }

    public b.j.j.x.b a(int i) {
        return null;
    }

    public b.j.j.x.b b(int i) {
        return null;
    }

    public boolean c(int i, int i2, Bundle bundle) {
        return false;
    }

    public c(Object obj) {
        this.f2272a = obj;
    }
}