package b.h.c;

import android.content.Context;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.util.Log;
import android.util.SparseArray;
import android.util.Xml;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.ConstraintLayout;
import b.h.c.d;
import java.io.IOException;
import java.util.ArrayList;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: ConstraintLayoutStates.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public final ConstraintLayout f1950a;

    /* renamed from: b  reason: collision with root package name */
    public int f1951b = -1;

    /* renamed from: c  reason: collision with root package name */
    public int f1952c = -1;

    /* renamed from: d  reason: collision with root package name */
    public SparseArray<a> f1953d = new SparseArray<>();

    /* renamed from: e  reason: collision with root package name */
    public SparseArray<d> f1954e = new SparseArray<>();

    /* compiled from: ConstraintLayoutStates.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public int f1955a;

        /* renamed from: b  reason: collision with root package name */
        public ArrayList<b> f1956b = new ArrayList<>();

        /* renamed from: c  reason: collision with root package name */
        public int f1957c;

        /* renamed from: d  reason: collision with root package name */
        public d f1958d;

        public a(Context context, XmlPullParser xmlPullParser) {
            this.f1957c = -1;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(Xml.asAttributeSet(xmlPullParser), i.f2016h);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 0) {
                    this.f1955a = obtainStyledAttributes.getResourceId(index, this.f1955a);
                } else if (index == 1) {
                    this.f1957c = obtainStyledAttributes.getResourceId(index, this.f1957c);
                    String resourceTypeName = context.getResources().getResourceTypeName(this.f1957c);
                    context.getResources().getResourceName(this.f1957c);
                    if ("layout".equals(resourceTypeName)) {
                        d dVar = new d();
                        this.f1958d = dVar;
                        dVar.c((ConstraintLayout) LayoutInflater.from(context).inflate(this.f1957c, (ViewGroup) null));
                    }
                }
            }
            obtainStyledAttributes.recycle();
        }

        public int a(float f2, float f3) {
            for (int i = 0; i < this.f1956b.size(); i++) {
                if (this.f1956b.get(i).a(f2, f3)) {
                    return i;
                }
            }
            return -1;
        }
    }

    /* compiled from: ConstraintLayoutStates.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public float f1959a;

        /* renamed from: b  reason: collision with root package name */
        public float f1960b;

        /* renamed from: c  reason: collision with root package name */
        public float f1961c;

        /* renamed from: d  reason: collision with root package name */
        public float f1962d;

        /* renamed from: e  reason: collision with root package name */
        public int f1963e;

        /* renamed from: f  reason: collision with root package name */
        public d f1964f;

        public b(Context context, XmlPullParser xmlPullParser) {
            this.f1959a = Float.NaN;
            this.f1960b = Float.NaN;
            this.f1961c = Float.NaN;
            this.f1962d = Float.NaN;
            this.f1963e = -1;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(Xml.asAttributeSet(xmlPullParser), i.j);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 0) {
                    this.f1963e = obtainStyledAttributes.getResourceId(index, this.f1963e);
                    String resourceTypeName = context.getResources().getResourceTypeName(this.f1963e);
                    context.getResources().getResourceName(this.f1963e);
                    if ("layout".equals(resourceTypeName)) {
                        d dVar = new d();
                        this.f1964f = dVar;
                        dVar.c((ConstraintLayout) LayoutInflater.from(context).inflate(this.f1963e, (ViewGroup) null));
                    }
                } else if (index == 1) {
                    this.f1962d = obtainStyledAttributes.getDimension(index, this.f1962d);
                } else if (index == 2) {
                    this.f1960b = obtainStyledAttributes.getDimension(index, this.f1960b);
                } else if (index == 3) {
                    this.f1961c = obtainStyledAttributes.getDimension(index, this.f1961c);
                } else if (index == 4) {
                    this.f1959a = obtainStyledAttributes.getDimension(index, this.f1959a);
                } else {
                    Log.v("ConstraintLayoutStates", "Unknown tag");
                }
            }
            obtainStyledAttributes.recycle();
        }

        public boolean a(float f2, float f3) {
            if (Float.isNaN(this.f1959a) || f2 >= this.f1959a) {
                if (Float.isNaN(this.f1960b) || f3 >= this.f1960b) {
                    if (Float.isNaN(this.f1961c) || f2 <= this.f1961c) {
                        return Float.isNaN(this.f1962d) || f3 <= this.f1962d;
                    }
                    return false;
                }
                return false;
            }
            return false;
        }
    }

    public c(Context context, ConstraintLayout constraintLayout, int i) {
        boolean z;
        this.f1950a = constraintLayout;
        XmlResourceParser xml = context.getResources().getXml(i);
        try {
            a aVar = null;
            for (int eventType = xml.getEventType(); eventType != 1; eventType = xml.next()) {
                if (eventType == 0) {
                    xml.getName();
                    continue;
                } else if (eventType != 2) {
                    continue;
                } else {
                    String name = xml.getName();
                    switch (name.hashCode()) {
                        case -1349929691:
                            if (name.equals("ConstraintSet")) {
                                z = true;
                                break;
                            }
                            z = true;
                            break;
                        case 80204913:
                            if (name.equals("State")) {
                                z = true;
                                break;
                            }
                            z = true;
                            break;
                        case 1382829617:
                            if (name.equals("StateSet")) {
                                z = true;
                                break;
                            }
                            z = true;
                            break;
                        case 1657696882:
                            if (name.equals("layoutDescription")) {
                                z = false;
                                break;
                            }
                            z = true;
                            break;
                        case 1901439077:
                            if (name.equals("Variant")) {
                                z = true;
                                break;
                            }
                            z = true;
                            break;
                        default:
                            z = true;
                            break;
                    }
                    if (z && !z) {
                        if (z) {
                            a aVar2 = new a(context, xml);
                            this.f1953d.put(aVar2.f1955a, aVar2);
                            aVar = aVar2;
                            continue;
                        } else if (z) {
                            b bVar = new b(context, xml);
                            if (aVar != null) {
                                aVar.f1956b.add(bVar);
                                continue;
                            } else {
                                continue;
                            }
                        } else if (!z) {
                            Log.v("ConstraintLayoutStates", "unknown tag " + name);
                            continue;
                        } else {
                            a(context, xml);
                            continue;
                        }
                    }
                }
            }
        } catch (IOException e2) {
            e2.printStackTrace();
        } catch (XmlPullParserException e3) {
            e3.printStackTrace();
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:114:0x01d0, code lost:
        continue;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void a(Context context, XmlPullParser xmlPullParser) {
        int eventType;
        d.a aVar;
        d.a e2;
        d dVar = new d();
        int attributeCount = xmlPullParser.getAttributeCount();
        for (int i = 0; i < attributeCount; i++) {
            if ("id".equals(xmlPullParser.getAttributeName(i))) {
                String attributeValue = xmlPullParser.getAttributeValue(i);
                int identifier = attributeValue.contains("/") ? context.getResources().getIdentifier(attributeValue.substring(attributeValue.indexOf(47) + 1), "id", context.getPackageName()) : -1;
                if (identifier == -1) {
                    if (attributeValue.length() > 1) {
                        identifier = Integer.parseInt(attributeValue.substring(1));
                    } else {
                        Log.e("ConstraintLayoutStates", "error in parsing id");
                    }
                }
                try {
                    eventType = xmlPullParser.getEventType();
                    aVar = null;
                } catch (IOException e3) {
                    e3.printStackTrace();
                } catch (XmlPullParserException e4) {
                    e4.printStackTrace();
                }
                while (eventType != 1) {
                    if (eventType != 0) {
                        char c2 = 3;
                        if (eventType == 2) {
                            String name = xmlPullParser.getName();
                            switch (name.hashCode()) {
                                case -2025855158:
                                    if (name.equals("Layout")) {
                                        c2 = 5;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case -1984451626:
                                    if (name.equals("Motion")) {
                                        c2 = 6;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case -1269513683:
                                    if (name.equals("PropertySet")) {
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case -1238332596:
                                    if (name.equals("Transform")) {
                                        c2 = 4;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case -71750448:
                                    if (name.equals("Guideline")) {
                                        c2 = 1;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 1331510167:
                                    if (name.equals("Barrier")) {
                                        c2 = 2;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 1791837707:
                                    if (name.equals("CustomAttribute")) {
                                        c2 = 7;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 1803088381:
                                    if (name.equals("Constraint")) {
                                        c2 = 0;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                default:
                                    c2 = 65535;
                                    break;
                            }
                            switch (c2) {
                                case 0:
                                    e2 = dVar.e(context, Xml.asAttributeSet(xmlPullParser));
                                    aVar = e2;
                                    break;
                                case 1:
                                    e2 = dVar.e(context, Xml.asAttributeSet(xmlPullParser));
                                    d.b bVar = e2.f1973d;
                                    bVar.f1977b = true;
                                    bVar.f1978c = true;
                                    aVar = e2;
                                    break;
                                case 2:
                                    e2 = dVar.e(context, Xml.asAttributeSet(xmlPullParser));
                                    e2.f1973d.e0 = 1;
                                    aVar = e2;
                                    break;
                                case 3:
                                    if (aVar != null) {
                                        aVar.f1971b.a(context, Xml.asAttributeSet(xmlPullParser));
                                        break;
                                    } else {
                                        throw new RuntimeException("XML parser error must be within a Constraint " + xmlPullParser.getLineNumber());
                                    }
                                case 4:
                                    if (aVar != null) {
                                        aVar.f1974e.a(context, Xml.asAttributeSet(xmlPullParser));
                                        break;
                                    } else {
                                        throw new RuntimeException("XML parser error must be within a Constraint " + xmlPullParser.getLineNumber());
                                    }
                                case 5:
                                    if (aVar != null) {
                                        aVar.f1973d.a(context, Xml.asAttributeSet(xmlPullParser));
                                        break;
                                    } else {
                                        throw new RuntimeException("XML parser error must be within a Constraint " + xmlPullParser.getLineNumber());
                                    }
                                case 6:
                                    if (aVar != null) {
                                        aVar.f1972c.a(context, Xml.asAttributeSet(xmlPullParser));
                                        break;
                                    } else {
                                        throw new RuntimeException("XML parser error must be within a Constraint " + xmlPullParser.getLineNumber());
                                    }
                                case 7:
                                    if (aVar != null) {
                                        b.h.c.a.a(context, xmlPullParser, aVar.f1975f);
                                        break;
                                    } else {
                                        throw new RuntimeException("XML parser error must be within a Constraint " + xmlPullParser.getLineNumber());
                                    }
                            }
                        } else if (eventType != 3) {
                            continue;
                        } else {
                            String name2 = xmlPullParser.getName();
                            if (!"ConstraintSet".equals(name2)) {
                                if (name2.equalsIgnoreCase("Constraint")) {
                                    dVar.f1969e.put(Integer.valueOf(aVar.f1970a), aVar);
                                    aVar = null;
                                }
                            } else {
                                this.f1954e.put(identifier, dVar);
                                return;
                            }
                        }
                    } else {
                        xmlPullParser.getName();
                    }
                    eventType = xmlPullParser.next();
                }
                this.f1954e.put(identifier, dVar);
                return;
            }
        }
    }
}