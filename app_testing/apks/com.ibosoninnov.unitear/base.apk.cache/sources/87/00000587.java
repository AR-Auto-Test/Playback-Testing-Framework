package b.v;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.os.Bundle;
import android.os.Parcelable;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.util.Xml;
import b.v.p;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: NavInflater.java */
/* loaded from: classes.dex */
public final class n {

    /* renamed from: a  reason: collision with root package name */
    public static final ThreadLocal<TypedValue> f2659a = new ThreadLocal<>();

    /* renamed from: b  reason: collision with root package name */
    public Context f2660b;

    /* renamed from: c  reason: collision with root package name */
    public r f2661c;

    public n(Context context, r rVar) {
        this.f2660b = context;
        this.f2661c = rVar;
    }

    public static p a(TypedValue typedValue, p pVar, p pVar2, String str, String str2) {
        if (pVar == null || pVar == pVar2) {
            return pVar != null ? pVar : pVar2;
        }
        throw new XmlPullParserException("Type is " + str + " but found " + str2 + ": " + typedValue.data);
    }

    /* JADX WARN: Code restructure failed: missing block: B:100:0x022a, code lost:
        return r3;
     */
    /* JADX WARN: Code restructure failed: missing block: B:80:0x01a2, code lost:
        r16 = r4;
        r5.isEmpty();
        r5 = 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:81:0x01ab, code lost:
        if ((!(r3 instanceof b.v.a.C0050a)) == false) goto L78;
     */
    /* JADX WARN: Code restructure failed: missing block: B:82:0x01ad, code lost:
        if (r13 == 0) goto L75;
     */
    /* JADX WARN: Code restructure failed: missing block: B:84:0x01b1, code lost:
        if (r3.f2649h != null) goto L74;
     */
    /* JADX WARN: Code restructure failed: missing block: B:85:0x01b3, code lost:
        r3.f2649h = new b.f.i<>(10);
     */
    /* JADX WARN: Code restructure failed: missing block: B:86:0x01bc, code lost:
        r3.f2649h.g(r13, r15);
        r6.recycle();
     */
    /* JADX WARN: Code restructure failed: missing block: B:88:0x01cc, code lost:
        throw new java.lang.IllegalArgumentException("Cannot have an action with actionId 0");
     */
    /* JADX WARN: Code restructure failed: missing block: B:90:0x01f0, code lost:
        throw new java.lang.UnsupportedOperationException("Cannot add action " + r13 + " to " + r3 + " as it does not support actions, indicating that it is a terminal destination in your navigation graph and will never trigger actions.");
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final j b(Resources resources, XmlResourceParser xmlResourceParser, AttributeSet attributeSet, int i) {
        int i2;
        int depth;
        int i3;
        int i4;
        String str;
        j a2 = this.f2661c.c(xmlResourceParser.getName()).a();
        a2.d(this.f2660b, attributeSet);
        int i5 = 1;
        int depth2 = xmlResourceParser.getDepth() + 1;
        while (true) {
            int next = xmlResourceParser.next();
            if (next == i5) {
                break;
            }
            int depth3 = xmlResourceParser.getDepth();
            int i6 = 3;
            if (depth3 < depth2 && next == 3) {
                break;
            } else if (next == 2 && depth3 <= depth2) {
                String name = xmlResourceParser.getName();
                if ("argument".equals(name)) {
                    TypedArray obtainAttributes = resources.obtainAttributes(attributeSet, b.v.t.a.f2683b);
                    String string = obtainAttributes.getString(0);
                    if (string != null) {
                        d d2 = d(obtainAttributes, resources, i);
                        if (a2.i == null) {
                            a2.i = new HashMap<>();
                        }
                        a2.i.put(string, d2);
                        obtainAttributes.recycle();
                    } else {
                        throw new XmlPullParserException("Arguments must have a name");
                    }
                } else if ("deepLink".equals(name)) {
                    TypedArray obtainAttributes2 = resources.obtainAttributes(attributeSet, b.v.t.a.f2684c);
                    String string2 = obtainAttributes2.getString(3);
                    String string3 = obtainAttributes2.getString(i5);
                    String string4 = obtainAttributes2.getString(2);
                    if (TextUtils.isEmpty(string2) && TextUtils.isEmpty(string3) && TextUtils.isEmpty(string4)) {
                        throw new XmlPullParserException("Every <deepLink> must include at least one of app:uri, app:action, or app:mimeType");
                    }
                    String replace = string2 != null ? string2.replace("${applicationId}", this.f2660b.getPackageName()) : null;
                    if (TextUtils.isEmpty(string3)) {
                        str = null;
                    } else {
                        str = string3.replace("${applicationId}", this.f2660b.getPackageName());
                        if (str.isEmpty()) {
                            throw new IllegalArgumentException("The NavDeepLink cannot have an empty action.");
                        }
                    }
                    h hVar = new h(replace, str, string4 != null ? string4.replace("${applicationId}", this.f2660b.getPackageName()) : null);
                    if (a2.f2648g == null) {
                        a2.f2648g = new ArrayList<>();
                    }
                    a2.f2648g.add(hVar);
                    obtainAttributes2.recycle();
                } else {
                    if ("action".equals(name)) {
                        TypedArray obtainAttributes3 = resources.obtainAttributes(attributeSet, b.v.t.a.f2682a);
                        int resourceId = obtainAttributes3.getResourceId(0, 0);
                        c cVar = new c(obtainAttributes3.getResourceId(i5, 0));
                        obtainAttributes3.getBoolean(4, false);
                        obtainAttributes3.getResourceId(7, -1);
                        obtainAttributes3.getBoolean(8, false);
                        obtainAttributes3.getResourceId(2, -1);
                        obtainAttributes3.getResourceId(3, -1);
                        obtainAttributes3.getResourceId(5, -1);
                        obtainAttributes3.getResourceId(6, -1);
                        Bundle bundle = new Bundle();
                        int i7 = 1;
                        int depth4 = xmlResourceParser.getDepth() + 1;
                        int i8 = i;
                        while (true) {
                            int next2 = xmlResourceParser.next();
                            if (next2 == i7 || ((depth = xmlResourceParser.getDepth()) < depth4 && next2 == i6)) {
                                break;
                            }
                            if (next2 == 2 && depth <= depth4) {
                                if ("argument".equals(xmlResourceParser.getName())) {
                                    TypedArray obtainAttributes4 = resources.obtainAttributes(attributeSet, b.v.t.a.f2683b);
                                    String string5 = obtainAttributes4.getString(0);
                                    if (string5 != null) {
                                        d d3 = d(obtainAttributes4, resources, i8);
                                        i3 = depth2;
                                        boolean z = d3.f2613c;
                                        if (z && z) {
                                            d3.f2611a.d(bundle, string5, d3.f2614d);
                                        }
                                        obtainAttributes4.recycle();
                                    } else {
                                        throw new XmlPullParserException("Arguments must have a name");
                                    }
                                } else {
                                    i3 = depth2;
                                }
                                i4 = i;
                            } else {
                                int i9 = i8;
                                i3 = depth2;
                                i4 = i9;
                            }
                            i6 = 3;
                            i7 = 1;
                            int i10 = i3;
                            i8 = i4;
                            depth2 = i10;
                        }
                    } else {
                        i2 = depth2;
                        if ("include".equals(name) && (a2 instanceof k)) {
                            TypedArray obtainAttributes5 = resources.obtainAttributes(attributeSet, s.f2681c);
                            ((k) a2).e(c(obtainAttributes5.getResourceId(0, 0)));
                            obtainAttributes5.recycle();
                        } else if (a2 instanceof k) {
                            ((k) a2).e(b(resources, xmlResourceParser, attributeSet, i));
                        }
                    }
                    depth2 = i2;
                }
                i2 = depth2;
                depth2 = i2;
            }
        }
    }

    @SuppressLint({"ResourceType"})
    public k c(int i) {
        int next;
        Resources resources = this.f2660b.getResources();
        XmlResourceParser xml = resources.getXml(i);
        AttributeSet asAttributeSet = Xml.asAttributeSet(xml);
        while (true) {
            try {
                try {
                    next = xml.next();
                    if (next == 2 || next == 1) {
                        break;
                    }
                } catch (Exception e2) {
                    throw new RuntimeException("Exception inflating " + resources.getResourceName(i) + " line " + xml.getLineNumber(), e2);
                }
            } finally {
                xml.close();
            }
        }
        if (next == 2) {
            String name = xml.getName();
            j b2 = b(resources, xml, asAttributeSet, i);
            if (b2 instanceof k) {
                return (k) b2;
            }
            throw new IllegalArgumentException("Root element <" + name + "> did not inflate into a NavGraph");
        }
        throw new XmlPullParserException("No start tag found");
    }

    public final d d(TypedArray typedArray, Resources resources, int i) {
        p<Integer> pVar;
        Object obj;
        p pVar2;
        p pVar3;
        String str;
        p c0051p;
        boolean z = typedArray.getBoolean(3, false);
        ThreadLocal<TypedValue> threadLocal = f2659a;
        TypedValue typedValue = threadLocal.get();
        if (typedValue == null) {
            typedValue = new TypedValue();
            threadLocal.set(typedValue);
        }
        String string = typedArray.getString(2);
        if (string != null) {
            String resourcePackageName = resources.getResourcePackageName(i);
            pVar = p.f2669a;
            if (!"integer".equals(string)) {
                pVar = p.f2671c;
                if (!"integer[]".equals(string)) {
                    pVar = p.f2672d;
                    if (!"long".equals(string)) {
                        pVar = p.f2673e;
                        if (!"long[]".equals(string)) {
                            pVar = p.f2676h;
                            if (!"boolean".equals(string)) {
                                pVar = p.i;
                                if (!"boolean[]".equals(string)) {
                                    pVar = p.j;
                                    if (!"string".equals(string)) {
                                        p pVar4 = p.k;
                                        if (!"string[]".equals(string)) {
                                            pVar4 = p.f2674f;
                                            if (!"float".equals(string)) {
                                                pVar4 = p.f2675g;
                                                if (!"float[]".equals(string)) {
                                                    pVar4 = p.f2670b;
                                                    if (!"reference".equals(string)) {
                                                        if (!string.isEmpty()) {
                                                            try {
                                                                if (!string.startsWith(".") || resourcePackageName == null) {
                                                                    str = string;
                                                                } else {
                                                                    str = resourcePackageName + string;
                                                                }
                                                                if (string.endsWith("[]")) {
                                                                    str = str.substring(0, str.length() - 2);
                                                                    Class<?> cls = Class.forName(str);
                                                                    if (Parcelable.class.isAssignableFrom(cls)) {
                                                                        c0051p = new p.m(cls);
                                                                    } else {
                                                                        if (Serializable.class.isAssignableFrom(cls)) {
                                                                            c0051p = new p.o(cls);
                                                                        }
                                                                        throw new IllegalArgumentException(str + " is not Serializable or Parcelable.");
                                                                    }
                                                                    pVar = c0051p;
                                                                } else {
                                                                    Class<?> cls2 = Class.forName(str);
                                                                    if (Parcelable.class.isAssignableFrom(cls2)) {
                                                                        c0051p = new p.n(cls2);
                                                                    } else if (Enum.class.isAssignableFrom(cls2)) {
                                                                        c0051p = new p.l(cls2);
                                                                    } else {
                                                                        if (Serializable.class.isAssignableFrom(cls2)) {
                                                                            c0051p = new p.C0051p(cls2);
                                                                        }
                                                                        throw new IllegalArgumentException(str + " is not Serializable or Parcelable.");
                                                                    }
                                                                    pVar = c0051p;
                                                                }
                                                            } catch (ClassNotFoundException e2) {
                                                                throw new RuntimeException(e2);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        pVar = pVar4;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            pVar = null;
        }
        boolean z2 = true;
        if (typedArray.getValue(1, typedValue)) {
            p<Integer> pVar5 = p.f2670b;
            if (pVar == pVar5) {
                int i2 = typedValue.resourceId;
                if (i2 != 0) {
                    obj = Integer.valueOf(i2);
                } else if (typedValue.type == 16 && typedValue.data == 0) {
                    obj = 0;
                } else {
                    StringBuilder x = c.b.a.a.a.x("unsupported value '");
                    x.append((Object) typedValue.string);
                    x.append("' for ");
                    x.append(pVar.b());
                    x.append(". Must be a reference to a resource.");
                    throw new XmlPullParserException(x.toString());
                }
            } else {
                int i3 = typedValue.resourceId;
                if (i3 != 0) {
                    if (pVar == null) {
                        obj = Integer.valueOf(i3);
                        pVar = pVar5;
                    } else {
                        StringBuilder x2 = c.b.a.a.a.x("unsupported value '");
                        x2.append((Object) typedValue.string);
                        x2.append("' for ");
                        x2.append(pVar.b());
                        x2.append(". You must use a \"");
                        throw new XmlPullParserException(c.b.a.a.a.v(x2, "reference", "\" type to reference other resources."));
                    }
                } else if (pVar == p.j) {
                    obj = typedArray.getString(1);
                } else {
                    int i4 = typedValue.type;
                    if (i4 == 3) {
                        String charSequence = typedValue.string.toString();
                        if (pVar == null) {
                            try {
                                try {
                                    try {
                                        try {
                                            pVar3 = p.f2669a;
                                            pVar3.c(charSequence);
                                        } catch (IllegalArgumentException unused) {
                                            pVar3 = p.f2676h;
                                            pVar3.c(charSequence);
                                        }
                                    } catch (IllegalArgumentException unused2) {
                                        pVar3 = p.f2672d;
                                        pVar3.c(charSequence);
                                    }
                                } catch (IllegalArgumentException unused3) {
                                    pVar3 = p.j;
                                }
                            } catch (IllegalArgumentException unused4) {
                                pVar3 = p.f2674f;
                                pVar3.c(charSequence);
                            }
                            pVar = pVar3;
                        }
                        obj = pVar.c(charSequence);
                    } else if (i4 == 4) {
                        pVar = a(typedValue, pVar, p.f2674f, string, "float");
                        obj = Float.valueOf(typedValue.getFloat());
                    } else if (i4 == 5) {
                        pVar = a(typedValue, pVar, p.f2669a, string, "dimension");
                        obj = Integer.valueOf((int) typedValue.getDimension(resources.getDisplayMetrics()));
                    } else if (i4 == 18) {
                        pVar = a(typedValue, pVar, p.f2676h, string, "boolean");
                        obj = Boolean.valueOf(typedValue.data != 0);
                    } else if (i4 >= 16 && i4 <= 31) {
                        p<Float> pVar6 = p.f2674f;
                        if (pVar == pVar6) {
                            pVar = a(typedValue, pVar, pVar6, string, "float");
                            obj = Float.valueOf(typedValue.data);
                        } else {
                            pVar = a(typedValue, pVar, p.f2669a, string, "integer");
                            obj = Integer.valueOf(typedValue.data);
                        }
                    } else {
                        StringBuilder x3 = c.b.a.a.a.x("unsupported argument type ");
                        x3.append(typedValue.type);
                        throw new XmlPullParserException(x3.toString());
                    }
                }
            }
        } else {
            obj = null;
        }
        if (obj == null) {
            obj = null;
            z2 = false;
        }
        if (pVar == null) {
            pVar = null;
        }
        if (pVar == null) {
            if (obj instanceof Integer) {
                pVar2 = p.f2669a;
            } else if (obj instanceof int[]) {
                pVar2 = p.f2671c;
            } else if (obj instanceof Long) {
                pVar2 = p.f2672d;
            } else if (obj instanceof long[]) {
                pVar2 = p.f2673e;
            } else if (obj instanceof Float) {
                pVar2 = p.f2674f;
            } else if (obj instanceof float[]) {
                pVar2 = p.f2675g;
            } else if (obj instanceof Boolean) {
                pVar2 = p.f2676h;
            } else if (obj instanceof boolean[]) {
                pVar2 = p.i;
            } else if (!(obj instanceof String) && obj != null) {
                if (obj instanceof String[]) {
                    pVar2 = p.k;
                } else if (obj.getClass().isArray() && Parcelable.class.isAssignableFrom(obj.getClass().getComponentType())) {
                    pVar2 = new p.m(obj.getClass().getComponentType());
                } else if (obj.getClass().isArray() && Serializable.class.isAssignableFrom(obj.getClass().getComponentType())) {
                    pVar2 = new p.o(obj.getClass().getComponentType());
                } else if (obj instanceof Parcelable) {
                    pVar2 = new p.n(obj.getClass());
                } else if (obj instanceof Enum) {
                    pVar2 = new p.l(obj.getClass());
                } else if (obj instanceof Serializable) {
                    pVar2 = new p.C0051p(obj.getClass());
                } else {
                    StringBuilder x4 = c.b.a.a.a.x("Object of type ");
                    x4.append(obj.getClass().getName());
                    x4.append(" is not supported for navigation arguments.");
                    throw new IllegalArgumentException(x4.toString());
                }
            } else {
                pVar2 = p.j;
            }
            pVar = pVar2;
        }
        return new d(pVar, z, obj, z2);
    }
}