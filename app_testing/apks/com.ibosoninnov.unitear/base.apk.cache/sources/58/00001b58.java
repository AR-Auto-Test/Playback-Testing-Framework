package com.google.android.play.core.splitinstall;

import android.util.Log;
import java.io.IOException;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzbg {
    public static final zzk zza(XmlPullParser xmlPullParser, zzi zziVar) {
        while (xmlPullParser.next() != 1) {
            try {
                if (xmlPullParser.getEventType() == 2) {
                    if (xmlPullParser.getName().equals("splits")) {
                        while (xmlPullParser.next() != 3) {
                            if (xmlPullParser.getEventType() == 2) {
                                if (xmlPullParser.getName().equals("module")) {
                                    String zzb = zzb("name", xmlPullParser, zziVar);
                                    if (zzb != null) {
                                        while (xmlPullParser.next() != 3) {
                                            if (xmlPullParser.getEventType() == 2) {
                                                if (xmlPullParser.getName().equals("language")) {
                                                    while (xmlPullParser.next() != 3) {
                                                        if (xmlPullParser.getEventType() == 2) {
                                                            if (xmlPullParser.getName().equals("entry")) {
                                                                String zzb2 = zzb("key", xmlPullParser, zziVar);
                                                                String zzb3 = zzb("split", xmlPullParser, zziVar);
                                                                zzc(xmlPullParser, zziVar);
                                                                if (zzb2 != null && zzb3 != null) {
                                                                    zziVar.zza(zzb, zzb2, zzb3);
                                                                }
                                                            } else {
                                                                zzc(xmlPullParser, zziVar);
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    zzc(xmlPullParser, zziVar);
                                                }
                                            }
                                        }
                                    } else {
                                        zzc(xmlPullParser, zziVar);
                                    }
                                } else {
                                    zzc(xmlPullParser, zziVar);
                                }
                            }
                        }
                    } else {
                        zzc(xmlPullParser, zziVar);
                    }
                }
            } catch (IOException | IllegalStateException | XmlPullParserException e2) {
                Log.e("SplitInstall", "Error while parsing splits.xml", e2);
                return null;
            }
        }
        return zziVar.zzb();
    }

    private static final String zzb(String str, XmlPullParser xmlPullParser, zzi zziVar) {
        for (int i = 0; i < xmlPullParser.getAttributeCount(); i++) {
            if (xmlPullParser.getAttributeName(i).equals(str)) {
                return xmlPullParser.getAttributeValue(i);
            }
        }
        return null;
    }

    private static final void zzc(XmlPullParser xmlPullParser, zzi zziVar) {
        int i = 1;
        while (i != 0) {
            int next = xmlPullParser.next();
            if (next == 2) {
                i++;
            } else if (next == 3) {
                i--;
            }
        }
    }
}