// ReportPDF.jsx
import React from 'react';
import { Document, Page, Text, View, StyleSheet, Image } from '@react-pdf/renderer';

// Define styles
const styles = StyleSheet.create({
  page: {
    padding: 40,
    fontSize: 12,
    fontFamily: 'Helvetica',
  },
  header: {
    marginBottom: 20,
    borderBottom: '1 solid #eee',
    paddingBottom: 10,
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
  },
  logo: {
    width: 50,
    height: 50,
    marginRight: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  section: {
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  text: {
    marginBottom: 2,
  },
});

const ReportPDF = ({ reportData }) => {
  const { symptoms = [], report = '', confidence = 0 } = reportData || {};
  const currentDate = new Date().toLocaleString();

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {/* Header with logo and title */}
        <View style={styles.header}>
          <Image
            style={styles.logo}
            src="https://yourdomain.com/logo.png" // Replace with your logo URL
          />
          <Text style={styles.title}>Diagnostic Report</Text>
        </View>

        {/* Report Summary */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <Text style={styles.text}>{report}</Text>
        </View>

        {/* Symptoms */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Symptoms</Text>
          {symptoms.map((symptom, index) => (
            <Text key={index} style={styles.text}>- {symptom}</Text>
          ))}
        </View>

        {/* Confidence */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Confidence Level</Text>
          <Text style={styles.text}>{confidence}%</Text>
        </View>

        {/* Date */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Date</Text>
          <Text style={styles.text}>{currentDate}</Text>
        </View>
      </Page>
    </Document>
  );
};

export default ReportPDF;
