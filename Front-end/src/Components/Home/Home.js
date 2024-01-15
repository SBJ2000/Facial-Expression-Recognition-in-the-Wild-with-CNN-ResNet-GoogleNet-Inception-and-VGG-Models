import React from "react";
import { Container, Row, Col , Button} from "react-bootstrap";
import homeLogo from "../../Assets/home-main.svg";
import { Link } from "react-router-dom";

const Home = () => {
  const handlePredictClickResnet = () => {
    // Redirect to another URL
    window.location.href = '/Resnet';
  };
  const handlePredictClickCNN = () => {
    // Redirect to another URL
    window.location.href = '/CNN';
  };
  const handlePredictClickInception = () => {
    // Redirect to another URL
    window.location.href = '/Inception';
  };
  const handlePredictClickGoogleNet = () => {
    // Redirect to another URL
    window.location.href = '/GoogleNet';
  };
  const handlePredictClickVgg = () => {
    // Redirect to another URL
    window.location.href = '/Vgg';
  };
  return (
    <section>
      <Container fluid className="home-section" id="home">
        <Container className="home-content">
          <Row>
            <Col md={7} className="home-header">
              <h1 style={{ paddingBottom: 15 }} className="heading">
                Welcome to The Emotion Recognition App{" "}
                <span className="wave" role="img" aria-labelledby="wave">
                  👋🏻
                </span>
              </h1>

              <h1 className="heading-name" style={{ color: '#c770f0' }}>
                Understand emotions like never before, using different technologies.
              </h1>

             
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <img
                src={homeLogo}
                alt="home pic"
                className="img-fluid"
                style={{ maxHeight: "450px" }}
              />
            </Col>
            
          </Row>
          
        </Container>
      
      </Container>  
      <div>
        <Button variant="primary" style={{ marginTop: "10px", marginRight: "5px" }} onClick={handlePredictClickResnet}>
          Predict Emotion with Resnet Model
        </Button>
        <Button variant="primary" style={{ marginTop: "10px", marginRight: "5px" }} onClick={handlePredictClickCNN}>
          Predict Emotion with CNN Model
        </Button>
        <Button variant="primary" style={{ marginTop: "10px" }} onClick={handlePredictClickInception}>
          Predict Emotion with Inception Model
        </Button>
      </div>
      <div>
        <Button variant="primary" style={{ marginTop: "10px", marginRight: "5px" }} onClick={handlePredictClickGoogleNet}>
          Predict Emotion with GoogleNet Model
        </Button>
        <Button variant="primary" style={{ marginTop: "10px" }} onClick={handlePredictClickVgg}>
          Predict Emotion with Vgg Model
        </Button>
      </div>
    </section>
  );
}

export default Home;